"""
Planning (DeepPlanning) Benchmark Runner.

Single-turn planning evaluation using Qwen/DeepPlanning-style tasks.
Scores via heuristic constraint-checking (no sandbox).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ....agents.base_agent import BaseAgent, EvaluationResult
from ....evaluation.cost_tracker import CostTracker
from ....utils.adaptive_limiter import ThreadSafeAdaptiveLimiter
from ....utils.trace import TraceCapture, QuestionTrace


# Weights for composite score (travel)
TRAVEL_STRUCTURE_WEIGHT = 0.4
TRAVEL_BUDGET_WEIGHT = 0.3
TRAVEL_CONSTRAINT_WEIGHT = 0.3

# Weights for composite score (shopping)
SHOPPING_JSON_WEIGHT = 0.4
SHOPPING_BUDGET_WEIGHT = 0.3
SHOPPING_COMPLETENESS_WEIGHT = 0.3

SUCCESS_THRESHOLD = 0.5


def _extract_budget_from_text(text: str) -> Optional[float]:
    """Extract a single budget amount (number, optionally with currency)."""
    # Match patterns like "3000 CNY", "¥3000", "$500", "budget: 800", "500 USD"
    patterns = [
        r"(?:budget|total|cap|max)\s*[:\s]*[\$¥€]?\s*([\d,]+(?:\.[\d]+)?)",
        r"[\$¥€]\s*([\d,]+(?:\.[\d]+)?)",
        r"(\d{2,})\s*(?:CNY|USD|EUR|GBP)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    # Last resort: any large number that looks like money
    numbers = re.findall(r"\b(\d{3,})\b", text)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return None


def _extract_travel_total_from_response(response: str) -> Optional[float]:
    """
    Extract the stated *total* cost from a travel itinerary response only when
    the model explicitly mentions a total (e.g. "Total: 2698", "Overall total: 2400 CNY").
    Avoids treating a line-item amount (e.g. "300 CNY" for one night) as the total.
    """
    if not response or not response.strip():
        return None
    text = response.strip().lower()
    # Prefer explicit total phrases (order matters: more specific first)
    patterns = [
        r"(?:overall\s+)?total\s*[:\s=]*[\$¥€]?\s*([\d,]+(?:\.[\d]+)?)\s*(?:cny|usd|eur|gbp)?",
        r"total\s+cost\s*[:\s=]*[\$¥€]?\s*([\d,]+(?:\.[\d]+)?)",
        r"budget\s+summary[^\d]*?(\d{3,})",
        r"(?:sum|grand total)\s*[:\s=]*[\$¥€]?\s*([\d,]+(?:\.[\d]+)?)",
        r"totaling\s+[\$¥€]?\s*([\d,]+(?:\.[\d]+)?)",
        r"=\s*([\d,]+(?:\.[\d]+)?)\s*(?:cny|usd)\s*(?:\)|\.|\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None


def _score_travel(
    response: str,
    query: str,
    constraints: List[str],
) -> Dict[str, float]:
    """Heuristic travel scoring: structure, budget, constraints."""
    response_lower = (response or "").strip().lower()
    query_lower = (query or "").lower()

    # Structure: day headers, budget summary, transport/hotel/meal/attraction
    has_day = bool(re.search(r"day\s*\d+", response_lower))
    has_budget_section = "budget" in response_lower or "total" in response_lower
    has_transport = any(
        x in response_lower for x in ["flight", "train", "transport", "travel"]
    )
    has_hotel = "hotel" in response_lower or "accommodation" in response_lower
    has_meal = any(x in response_lower for x in ["meal", "dinner", "lunch", "restaurant", "food"])
    has_attraction = any(
        x in response_lower for x in ["attraction", "visit", "sight", "tour"]
    )
    structure_checks = sum([has_day, has_budget_section, has_transport, has_hotel, has_meal, has_attraction])
    structure_score = min(1.0, structure_checks / 6.0) if structure_checks else 0.0

    # Budget: require explicit stated total in response (not a line-item amount)
    budget_cap = _extract_budget_from_text(query)
    stated_total = _extract_travel_total_from_response(response)
    if budget_cap is not None and stated_total is not None:
        budget_score = 1.0 if stated_total <= budget_cap * 1.01 else 0.0
    elif budget_cap is not None and stated_total is None:
        budget_score = 0.0  # cap given but model never stated a total
    else:
        budget_score = 0.5  # no cap to check

    # Constraint: destination and key constraint keywords in response
    if not constraints:
        constraint_score = 0.5
    else:
        found = sum(1 for c in constraints if c and c.lower() in response_lower)
        constraint_score = min(1.0, found / max(len(constraints), 1))
    # Also boost if query destination city appears
    city_like = re.findall(r"(?:to|in|from)\s+([A-Za-z\u4e00-\u9fff]+)", query)
    for c in city_like:
        if len(c) > 2 and c.lower() in response_lower:
            constraint_score = min(1.0, constraint_score + 0.2)
            break

    composite = (
        TRAVEL_STRUCTURE_WEIGHT * structure_score
        + TRAVEL_BUDGET_WEIGHT * budget_score
        + TRAVEL_CONSTRAINT_WEIGHT * constraint_score
    )
    return {
        "structure_score": structure_score,
        "budget_score": budget_score,
        "constraint_score": constraint_score,
        "composite_score": composite,
    }


def _score_shopping(
    response: str,
    query: str,
    budget_cap: Optional[float],
    required_item_count: int,
) -> Dict[str, float]:
    """Heuristic shopping scoring: valid JSON with items, budget, completeness."""
    response = (response or "").strip()

    # JSON with items array
    json_score = 0.0
    parsed: Optional[Dict[str, Any]] = None
    json_match = re.search(r"\{[\s\S]*\}", response)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed.get("items"), list):
                json_score = 1.0
        except json.JSONDecodeError:
            pass

    # Budget: final_total <= cap
    if budget_cap is not None and parsed is not None:
        total = None
        for key in ("final_total", "total", "finalTotal", "total_cost"):
            if key in parsed and isinstance(parsed[key], (int, float)):
                total = float(parsed[key])
                break
        if total is not None:
            budget_score = 1.0 if total <= budget_cap * 1.01 else 0.0
        else:
            budget_score = 0.0
    else:
        budget_score = 0.5

    # Completeness: item count >= required
    if parsed and isinstance(parsed.get("items"), list):
        item_count = len(parsed["items"])
        if required_item_count <= 0:
            completeness_score = 1.0
        else:
            completeness_score = min(1.0, item_count / required_item_count)
    else:
        completeness_score = 0.0

    composite = (
        SHOPPING_JSON_WEIGHT * json_score
        + SHOPPING_BUDGET_WEIGHT * budget_score
        + SHOPPING_COMPLETENESS_WEIGHT * completeness_score
    )
    return {
        "json_score": json_score,
        "budget_score": budget_score,
        "completeness_score": completeness_score,
        "composite_score": composite,
    }


class PlanningRunner:
    """
    Runner for Planning (DeepPlanning-style) evaluation.
    Single-turn: agent receives query and produces a plan; scored by heuristics.
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        cost_tracker: Optional[CostTracker] = None,
        verbose: bool = False,
        concurrency: int = 1,
        run_dir: Optional[Path] = None,
        domain: str = "all",
        language: str = "en",
    ):
        self.agent = agent
        self.cost_tracker = cost_tracker
        self.verbose = verbose
        self.concurrency = concurrency
        self.run_dir = run_dir
        self.domain = domain.lower() if domain else "all"
        self.language = language.lower() if language else "en"
        self._lock = Lock()
        self._output_dir: Optional[Path] = None
        self._total = 0
        self._success = 0
        self._scores: List[float] = []

    def load_tasks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load planning tasks from Qwen/DeepPlanning or synthetic fallback."""
        try:
            from datasets import load_dataset
        except Exception:
            return self._generate_synthetic_tasks(limit or 20)

        try:
            ds = load_dataset("Qwen/DeepPlanning", trust_remote_code=True)
            # Dataset may have splits like "travel", "shopping" or a single split
            if isinstance(ds, dict):
                all_rows: List[Dict[str, Any]] = []
                for split_name, subset in ds.items():
                    if self.domain != "all":
                        if self.domain not in split_name.lower():
                            continue
                    for i, row in enumerate(subset):
                        all_rows.append((split_name, i, row))
            else:
                all_rows = [(getattr(ds, "split", "train"), i, row) for i, row in enumerate(ds)]

            tasks: List[Dict[str, Any]] = []
            for split_name, idx, row in all_rows:
                domain_val = "travel" if "travel" in split_name.lower() else "shopping"
                if self.domain != "all" and self.domain != domain_val:
                    continue
                query = row.get("query") or row.get("question") or row.get("input") or ""
                if not query:
                    continue
                lang = (row.get("language") or row.get("lang") or "en").lower()
                if self.language != "all" and lang != self.language:
                    continue
                task_id = row.get("id") or row.get("task_id") or f"{domain_val}_{split_name}_{idx}"
                if isinstance(task_id, (int, float)):
                    task_id = f"{domain_val}_{task_id}"
                tasks.append({
                    "task_id": str(task_id),
                    "query": query,
                    "domain": domain_val,
                    "ground_truth": row.get("ground_truth") or row.get("reference") or {},
                    "constraints": row.get("constraints") or _extract_constraint_keywords(query),
                })
                if limit and len(tasks) >= limit:
                    break
            if limit:
                tasks = tasks[:limit]
            return tasks if tasks else self._generate_synthetic_tasks(limit or 20)
        except Exception:
            return self._generate_synthetic_tasks(limit or 20)

    def _generate_synthetic_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Generate synthetic travel and shopping tasks (50 unique examples)."""
        travel_examples = [
            {"query": "Plan a 3-day trip from Beijing to Chengdu, March 15-17. Budget 3000 CNY. 3-star hotel with washing machine. Flight after 7 AM. Visit Jinli Street.", "constraints": ["Chengdu", "3000", "3-star", "Jinli"]},
            {"query": "2-day trip from Beijing to Shanghai, budget 2000 CNY. Hotel near the Bund. Must visit the Oriental Pearl Tower and Yu Garden.", "constraints": ["Shanghai", "2000", "Bund", "Oriental Pearl", "Yu Garden"]},
            {"query": "4-day trip from Guangzhou to Xi'an, budget 4500 CNY. Want a 4-star hotel. Must include the Terracotta Warriors and the Ancient City Wall.", "constraints": ["Xi'an", "4500", "4-star", "Terracotta", "City Wall"]},
            {"query": "Plan a 5-day trip from Shanghai to Guilin, budget 5000 CNY. Prefer a boutique hotel near the Li River. Include a bamboo raft cruise and Yangshuo West Street.", "constraints": ["Guilin", "5000", "Li River", "Yangshuo"]},
            {"query": "3-day trip from Shenzhen to Chongqing, budget 2500 CNY. 3-star hotel. Meals budget max 100 CNY per day. Visit Hongyadong and Ciqikou.", "constraints": ["Chongqing", "2500", "Hongyadong", "Ciqikou"]},
            {"query": "2-day trip from Hangzhou to Nanjing, budget 1800 CNY. Hotel near Confucius Temple. Train departing before 9 AM. Visit Sun Yat-sen Mausoleum.", "constraints": ["Nanjing", "1800", "Confucius Temple", "Sun Yat-sen"]},
            {"query": "4-day trip from Chengdu to Lhasa, budget 6000 CNY. 3-star hotel. Must book altitude medication. Visit Potala Palace and Barkhor Street.", "constraints": ["Lhasa", "6000", "Potala", "Barkhor"]},
            {"query": "3-day couple's trip from Beijing to Qingdao, budget 3500 CNY. Beachside hotel. Must visit Zhanqiao Pier and Badaguan Scenic Area.", "constraints": ["Qingdao", "3500", "Zhanqiao", "Badaguan"]},
            {"query": "2-day family trip from Shanghai to Suzhou for 2 adults and 1 child, budget 2200 CNY. Hotel near Humble Administrator's Garden. Include Tongli Ancient Town.", "constraints": ["Suzhou", "2200", "Humble Administrator", "Tongli"]},
            {"query": "5-day solo trip from Beijing to Yunnan province, budget 4000 CNY. Hostel accommodation. Visit Dali Old Town, Lijiang, and Tiger Leaping Gorge.", "constraints": ["Yunnan", "4000", "Dali", "Lijiang", "Tiger Leaping"]},
            {"query": "3-day trip from Wuhan to Zhangjiajie, budget 2800 CNY. 3-star hotel. Train before 8 AM on day 1. Visit Avatar Hallelujah Mountain and Tianmen Mountain.", "constraints": ["Zhangjiajie", "2800", "Avatar", "Tianmen"]},
            {"query": "2-day business trip from Guangzhou to Shenzhen, budget 1500 CNY. 4-star hotel near Futian CBD. Need conference room access. Visit OCT Loft in the evening.", "constraints": ["Shenzhen", "1500", "4-star", "Futian", "OCT Loft"]},
            {"query": "4-day trip from Chongqing to Guizhou, budget 3200 CNY. 3-star hotel. Visit Huangguoshu Waterfall, Zhenyuan Ancient Town, and local Miao villages.", "constraints": ["Guizhou", "3200", "Huangguoshu", "Zhenyuan", "Miao"]},
            {"query": "3-day trip from Nanjing to Huangshan, budget 2600 CNY. Inn near the mountain entrance. Sunrise viewing at Guangming Peak. Visit Hongcun Village.", "constraints": ["Huangshan", "2600", "Guangming", "Hongcun"]},
            {"query": "5-day trip from Chengdu to Jiuzhaigou, budget 5500 CNY. Book in advance. Hotel in the valley. Must see Five Flower Lake and Nuorilang Waterfall.", "constraints": ["Jiuzhaigou", "5500", "Five Flower Lake", "Nuorilang"]},
            {"query": "2-day trip from Beijing to Tianjin, budget 1200 CNY. Hotel in the Italian Quarter. Visit Tianjin Eye, Ancient Culture Street, and local goubuli buns restaurant.", "constraints": ["Tianjin", "1200", "Tianjin Eye", "Culture Street"]},
            {"query": "3-day trip from Shanghai to Xiamen, budget 3000 CNY. Hotel on Gulangyu Island. No motor vehicles on the island. Visit Sunlight Rock and Piano Museum.", "constraints": ["Xiamen", "3000", "Gulangyu", "Sunlight Rock"]},
            {"query": "4-day self-drive trip from Chengdu to Tibet border, budget 4800 CNY. Rent a 4WD. Stay in guesthouses. Drive along Sichuan-Tibet Highway S303.", "constraints": ["Tibet", "4800", "4WD", "Sichuan-Tibet"]},
            {"query": "2-day trip from Hangzhou to Moganshan, budget 1600 CNY. Boutique mountain resort. Arrive by private car. Visit bamboo forests and old European villas.", "constraints": ["Moganshan", "1600", "bamboo", "villa"]},
            {"query": "3-day trip from Guangzhou to Zhangjiajie with senior parents (aged 65+), budget 3500 CNY. Wheelchair-accessible hotel. Avoid steep hikes. Cable car required.", "constraints": ["Zhangjiajie", "3500", "accessible", "cable car"]},
            {"query": "5-day photography trip from Beijing to Inner Mongolia, budget 4200 CNY. Camp on the grasslands. Rent a horse. Photograph the Gegentala prairie at sunrise.", "constraints": ["Inner Mongolia", "4200", "grassland", "horse", "sunrise"]},
            {"query": "2-day trip from Chengdu to Leshan, budget 1400 CNY. Hotel near the Giant Buddha. Boat tour at dawn to see the full statue. Visit Wuyou Temple.", "constraints": ["Leshan", "1400", "Giant Buddha", "boat", "Wuyou"]},
            {"query": "3-day trip from Xi'an to Zhangye, budget 2900 CNY. 3-star hotel. Visit Rainbow Mountains, Mati Temple, and Danxia Landform scenic area.", "constraints": ["Zhangye", "2900", "Rainbow Mountains", "Danxia"]},
            {"query": "4-day trip from Shanghai to Harbin in January, budget 4000 CNY. Heated hotel. International Ice and Snow Sculpture Festival. Siberian tiger park visit.", "constraints": ["Harbin", "4000", "Ice Festival", "tiger park"]},
            {"query": "2-day trip from Beijing to Datong, budget 1800 CNY. Hotel near Yungang Grottoes. Morning at the grottoes, afternoon at Hanging Monastery.", "constraints": ["Datong", "1800", "Yungang", "Hanging Monastery"]},
        ]
        shopping_examples = [
            {"query": "Buy running shoes (size 42, brand ShockWave, rating ≥4.7) and a matching sports jacket. Budget 800 CNY. Maximize coupon savings.", "constraints": ["running shoes", "size 42", "ShockWave", "800"], "required_items": 2},
            {"query": "Purchase a laptop under 5000 CNY and a wireless mouse. Total budget 5200 CNY. Prefer brands with good after-sales service.", "constraints": ["laptop", "5000", "wireless mouse", "5200"], "required_items": 2},
            {"query": "Buy a mirrorless camera body (Sony or Fujifilm, APS-C sensor) and a 35mm prime lens. Budget 8000 CNY total. Rating must be above 4.6.", "constraints": ["camera", "Sony", "Fujifilm", "lens", "8000"], "required_items": 2},
            {"query": "Order a robot vacuum cleaner (LiDAR navigation, max 3000 CNY) and a HEPA air purifier for a 50m² room, budget 1500 CNY. Delivery within 3 days.", "constraints": ["robot vacuum", "LiDAR", "3000", "air purifier", "1500"], "required_items": 2},
            {"query": "Purchase a standing desk (height 70-120cm, 130cm wide, max 2000 CNY) and an ergonomic office chair (lumbar support, max 1500 CNY). Total under 3500 CNY.", "constraints": ["standing desk", "2000", "ergonomic chair", "1500", "3500"], "required_items": 2},
            {"query": "Buy a camping tent (2-person, waterproof, under 600 CNY), a sleeping bag (rated to -5°C, under 400 CNY), and a camping stove (under 200 CNY). Total under 1200 CNY.", "constraints": ["tent", "2-person", "sleeping bag", "-5", "stove", "1200"], "required_items": 3},
            {"query": "Order a women's down jacket (size M, 90% duck down, black or navy, under 1200 CNY) and thermal leggings (size S, under 200 CNY). Fast delivery needed.", "constraints": ["down jacket", "size M", "90%", "1200", "thermal leggings"], "required_items": 2},
            {"query": "Buy noise-cancelling wireless headphones (40h battery, over-ear, under 1500 CNY) and a portable Bluetooth speaker (waterproof IPX7, under 400 CNY). Total under 1800 CNY.", "constraints": ["headphones", "noise-cancelling", "1500", "speaker", "IPX7", "400"], "required_items": 2},
            {"query": "Purchase a road bike (aluminum frame, Shimano 21-speed, under 3000 CNY) and a cycling helmet (size L, CPSC certified, under 300 CNY). Delivery to Shanghai.", "constraints": ["road bike", "Shimano", "3000", "helmet", "CPSC", "300"], "required_items": 2},
            {"query": "Buy a vitamin D3 supplement (2000 IU, 90 capsules, under 80 CNY), omega-3 fish oil (1000mg, 60 softgels, under 120 CNY), and a pill organizer. Total under 250 CNY.", "constraints": ["vitamin D3", "omega-3", "fish oil", "80", "120", "250"], "required_items": 3},
            {"query": "Order a 4K TV (55-inch, HDR, Dolby Atmos, under 3500 CNY) and a soundbar (2.1 channel, HDMI ARC, under 800 CNY). Brand preference: Sony or Samsung.", "constraints": ["4K TV", "55-inch", "3500", "soundbar", "HDMI ARC", "800"], "required_items": 2},
            {"query": "Buy a yoga mat (non-slip, 6mm thick, under 200 CNY), resistance bands set (5 levels, under 100 CNY), and foam roller (high density, under 150 CNY). Total under 400 CNY.", "constraints": ["yoga mat", "non-slip", "200", "resistance bands", "foam roller", "400"], "required_items": 3},
            {"query": "Purchase a mechanical keyboard (TKL, Cherry MX Red switches, RGB, under 500 CNY) and a gaming mouse (8000 DPI, programmable buttons, under 200 CNY). Total under 700 CNY.", "constraints": ["mechanical keyboard", "Cherry MX", "500", "gaming mouse", "8000 DPI", "700"], "required_items": 2},
            {"query": "Buy a baby stroller (lightweight, under 5kg, suitable for newborns, under 2000 CNY) and a car seat (group 0+, ECE R44 certified, under 1500 CNY). Safety rating must be highest tier.", "constraints": ["stroller", "lightweight", "2000", "car seat", "ECE R44", "1500"], "required_items": 2},
            {"query": "Order a French press coffee maker (1L, borosilicate glass, under 150 CNY), specialty coffee beans (Ethiopian single origin, 500g, under 100 CNY), and a gooseneck kettle (under 200 CNY). Total under 400 CNY.", "constraints": ["French press", "1L", "150", "Ethiopian", "coffee beans", "gooseneck kettle", "400"], "required_items": 3},
            {"query": "Purchase a men's formal suit (slim fit, size 48, navy blue, under 1500 CNY) and dress shoes (size 43, Oxford style, leather, under 600 CNY). Need before weekend.", "constraints": ["suit", "size 48", "navy", "1500", "Oxford", "size 43", "600"], "required_items": 2},
            {"query": "Buy a drawing tablet (A5 size, 8192 pressure levels, compatible with Photoshop, under 600 CNY) and a stylus pen set (fine tip, 0.3mm, 12 colors, under 80 CNY). Total under 700 CNY.", "constraints": ["drawing tablet", "A5", "8192", "600", "stylus", "0.3mm", "700"], "required_items": 2},
            {"query": "Order a portable power station (500Wh, pure sine wave, solar-compatible, under 2500 CNY) and a foldable solar panel (100W, under 800 CNY) for camping use.", "constraints": ["power station", "500Wh", "2500", "solar panel", "100W", "800"], "required_items": 2},
            {"query": "Buy a children's electric scooter (suitable age 6-12, max speed 15km/h, under 800 CNY) and a safety helmet (CE certified, size S, under 150 CNY). Total under 950 CNY.", "constraints": ["scooter", "age 6-12", "15km/h", "800", "helmet", "CE", "950"], "required_items": 2},
            {"query": "Purchase a smart watch (heart rate, SpO2, GPS, under 1200 CNY) and a replacement silicone band in black (compatible with the watch, under 50 CNY). Brand preference: Huawei or Garmin.", "constraints": ["smart watch", "heart rate", "GPS", "1200", "silicone band", "50"], "required_items": 2},
            {"query": "Buy a portable projector (1080p, 300 ANSI lumens, battery-powered, under 2000 CNY) and a 100-inch projector screen (foldable, under 300 CNY). Total under 2200 CNY.", "constraints": ["projector", "1080p", "300 ANSI", "2000", "screen", "100-inch", "2200"], "required_items": 2},
            {"query": "Order a sous vide cooker (1200W, precision ±0.1°C, under 300 CNY), a vacuum sealer (under 200 CNY), and a set of vacuum bags (100 count, under 50 CNY). Total under 500 CNY.", "constraints": ["sous vide", "1200W", "300", "vacuum sealer", "200", "vacuum bags", "500"], "required_items": 3},
            {"query": "Purchase a 3D printer (FDM, 220×220×250mm build volume, auto-leveling, under 1500 CNY) and 5 spools of PLA filament (1.75mm, assorted colors, under 50 CNY each). Total under 1800 CNY.", "constraints": ["3D printer", "FDM", "auto-leveling", "1500", "PLA filament", "1800"], "required_items": 2},
            {"query": "Buy a portable blender (USB rechargeable, BPA-free, 6 blades, under 150 CNY) and protein powder (whey isolate, chocolate flavor, 1kg, under 200 CNY). Total under 350 CNY.", "constraints": ["blender", "USB", "BPA-free", "150", "protein powder", "whey", "350"], "required_items": 2},
            {"query": "Order a smart home security camera (2K, outdoor, night vision, local storage, under 300 CNY) and a door/window sensor pack (4-piece, Zigbee protocol, under 150 CNY). Total under 400 CNY.", "constraints": ["security camera", "2K", "outdoor", "300", "door sensor", "Zigbee", "400"], "required_items": 2},
        ]
        tasks = []
        travel_idx = 0
        shopping_idx = 0
        for i in range(num_tasks):
            if self.domain == "shopping" or (self.domain == "all" and i % 2 == 1):
                ex = shopping_examples[shopping_idx % len(shopping_examples)]
                shopping_idx += 1
                tasks.append({
                    "task_id": f"synthetic_shopping_{i:04d}",
                    "query": ex["query"],
                    "domain": "shopping",
                    "ground_truth": {},
                    "constraints": ex.get("constraints", []),
                    "required_item_count": ex.get("required_items", 2),
                })
            else:
                ex = travel_examples[travel_idx % len(travel_examples)]
                travel_idx += 1
                tasks.append({
                    "task_id": f"synthetic_travel_{i:04d}",
                    "query": ex["query"],
                    "domain": "travel",
                    "ground_truth": {},
                    "constraints": ex.get("constraints", []),
                })
        if self.domain == "travel":
            return [t for t in tasks if t["domain"] == "travel"]
        if self.domain == "shopping":
            return [t for t in tasks if t["domain"] == "shopping"]
        return tasks

    def format_task(self, task: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Format a planning task for the agent."""
        query = task.get("query", "")
        domain = task.get("domain", "travel")

        if domain == "travel":
            task_text = f"""You are a travel planning assistant. Given the following request, produce a day-by-day itinerary with a clear budget summary at the end.

Request:
{query}

Provide:
1. Day-by-day schedule (Day 1, Day 2, ...) with times, transport, hotel, meals, and attractions.
2. A final "Budget Summary" section with line items and total cost (must not exceed the stated budget).

Return your plan as structured text (no JSON required)."""
        else:
            task_text = f"""You are a shopping planning assistant. Given the following request, produce a JSON shopping cart.

Request:
{query}

Provide a JSON object with:
- "items": array of chosen products (each with name, price, quantity as needed)
- "coupons": array of applied coupon codes (if any)
- "final_total": total cost after discounts (must not exceed the stated budget)

Return valid JSON only."""

        context = {
            "benchmark_type": "planning",
            "task_type": "planning",
            "domain": domain,
            "dataset": "deepplanning",
        }
        return task_text, context

    def _process_task(self, task: Dict[str, Any]) -> EvaluationResult:
        task_id = task["task_id"]
        task_text, context = self.format_task(task)
        query = task.get("query", "")
        domain = task.get("domain", "travel")
        constraints = task.get("constraints") or []

        with TraceCapture(
            task_id=task_id,
            agent_type=self.agent.__class__.__name__,
            model=self.agent.model,
            input_question=task_text[:1000] + "..." if len(task_text) > 1000 else task_text,
        ) as trace_ctx:
            start_time = time.time()
            try:
                response = self.agent.respond_to_task(task_text, context)
                latency = time.time() - start_time
                prediction = (response.response or "").strip()
            except Exception as e:
                latency = time.time() - start_time
                return EvaluationResult(
                    task_id=task_id,
                    prompt=task_text,
                    agent_response=f"Error: {e}",
                    success=False,
                    score=0.0,
                    latency=latency,
                    cost=0.0,
                    metadata={"error": str(e), "domain": domain},
                )

            if domain == "travel":
                scores = _score_travel(prediction, query, constraints)
            else:
                budget_cap = _extract_budget_from_text(query)
                required = task.get("required_item_count", 2)
                scores = _score_shopping(prediction, query, budget_cap, required)

            composite = scores.get("composite_score", 0.0)
            success = composite >= SUCCESS_THRESHOLD

            cost = 0.0
            meta = response.metadata if isinstance(response.metadata, dict) else {}
            if self.cost_tracker and meta:
                prompt_tokens = meta.get("prompt_tokens", 0)
                completion_tokens = meta.get("completion_tokens", 0)
                if prompt_tokens > 0:
                    with self._lock:
                        cost = self.cost_tracker.log_usage(
                            model=self.agent.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            task_id=task_id,
                            benchmark="planning",
                            agent_type=self.agent.__class__.__name__,
                        )

            trace_ctx.trace.final_output = prediction[:5000]
            trace_ctx.trace.predicted = prediction[:500]
            trace_ctx.trace.match = success
            trace_ctx.trace.confidence = composite
            trace_ctx.trace.total_latency = latency
            trace_ctx.trace.total_cost = cost
            trace_ctx.trace.reasoning = response.reasoning or ""

            return EvaluationResult(
                task_id=task_id,
                prompt=task_text[:1000] + "..." if len(task_text) > 1000 else task_text,
                agent_response=prediction,
                success=success,
                score=composite,
                latency=latency,
                cost=cost,
                metadata={
                    "domain": domain,
                    "query": query,
                    "constraints": constraints,
                    "trace": trace_ctx.trace,
                    **{k: v for k, v in scores.items() if k != "composite_score"},
                },
            )

    def _save_result_incremental(self, result: EvaluationResult) -> None:
        if not self._output_dir:
            return
        with self._lock:
            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            task_dir = self._output_dir / result.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            trace = metadata.get("trace")
            trace_path = task_dir / "trace.json"
            if trace and isinstance(trace, QuestionTrace):
                trace_path.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
            else:
                trace_path.write_text(
                    json.dumps(
                        {
                            "task_id": result.task_id,
                            "agent_type": self.agent.__class__.__name__,
                            "model": self.agent.model,
                            "predicted": (result.agent_response or "")[:500],
                            "match": bool(result.success),
                            "score": float(result.score or 0.0),
                            "total_latency": round(float(result.latency or 0.0), 2),
                            "total_cost": round(float(result.cost or 0.0), 6),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            summary_line = {
                "task_id": result.task_id,
                "score": round(float(result.score or 0.0), 6),
                "success": bool(result.success),
                "latency": round(float(result.latency or 0.0), 2),
                "cost": round(float(result.cost or 0.0), 6),
                "domain": metadata.get("domain"),
            }
            for k, v in metadata.items():
                if k not in ("trace", "query", "constraints") and isinstance(v, (int, float, str, bool)):
                    summary_line[k] = v
            with (self._output_dir / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary_line) + "\n")

    def _save_summary(self, results: List[EvaluationResult]) -> None:
        if not self._output_dir or not results:
            return
        total = len(results)
        success_count = sum(1 for r in results if r.success)
        travel_results = [r for r in results if (r.metadata or {}).get("domain") == "travel"]
        shopping_results = [r for r in results if (r.metadata or {}).get("domain") == "shopping"]
        mean_composite = sum(r.score or 0 for r in results) / total
        mean_latency = sum(r.latency or 0 for r in results) / total
        total_cost = sum(r.cost or 0 for r in results)

        # Sub-scores (travel: structure, budget, constraint; shopping: json, budget, completeness)
        structure_scores = []
        budget_scores = []
        constraint_scores = []
        for r in results:
            m = r.metadata or {}
            if m.get("domain") == "travel":
                structure_scores.append(m.get("structure_score"))
                budget_scores.append(m.get("budget_score"))
                constraint_scores.append(m.get("constraint_score"))
            else:
                structure_scores.append(m.get("json_score"))
                budget_scores.append(m.get("budget_score"))
                constraint_scores.append(m.get("completeness_score"))
        structure_scores = [x for x in structure_scores if x is not None]
        budget_scores = [x for x in budget_scores if x is not None]
        constraint_scores = [x for x in constraint_scores if x is not None]

        summary = {
            "agent": self.agent.__class__.__name__,
            "model": self.agent.model,
            "benchmark": "Planning (DeepPlanning)",
            "domain": self.domain,
            "language": self.language,
            "num_tasks": total,
            "mean_composite_score": round(mean_composite, 4),
            "mean_structure_score": round(sum(structure_scores) / len(structure_scores), 4) if structure_scores else 0,
            "mean_budget_score": round(sum(budget_scores) / len(budget_scores), 4) if budget_scores else 0,
            "mean_constraint_score": round(sum(constraint_scores) / len(constraint_scores), 4) if constraint_scores else 0,
            "success_rate": round(success_count / total, 4),
            "travel_success_rate": round(sum(1 for r in travel_results if r.success) / len(travel_results), 4) if travel_results else 0,
            "shopping_success_rate": round(sum(1 for r in shopping_results if r.success) / len(shopping_results), 4) if shopping_results else 0,
            "mean_latency": round(mean_latency, 2),
            "total_cost": round(total_cost, 4),
        }
        (self._output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if self.verbose:
            print(f"\nSummary saved to: {self._output_dir / 'summary.json'}")

    def _run_sequential(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        for task in tqdm(tasks, desc="Planning", disable=not self.verbose):
            result = self._process_task(task)
            results.append(result)
            with self._lock:
                self._total += 1
                self._success += 1 if result.success else 0
                self._scores.append(float(result.score or 0.0))
            self._save_result_incremental(result)
        return results

    def _run_concurrent(self, tasks: List[Dict[str, Any]]) -> List[EvaluationResult]:
        results: List[Optional[EvaluationResult]] = [None] * len(tasks)
        limiter = ThreadSafeAdaptiveLimiter(
            max_concurrency=self.concurrency,
            min_concurrency=1,
            backoff_factor=0.5,
            recovery_threshold=10,
        )

        def process_with_limiter(task: Dict[str, Any]) -> EvaluationResult:
            limiter.wait_if_needed()
            try:
                r = self._process_task(task)
                limiter.record_success()
                return r
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    limiter.record_error(backoff_seconds=5.0)
                raise

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(process_with_limiter, t): i for i, t in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = EvaluationResult(
                        task_id=tasks[idx].get("task_id", f"t_{idx}"),
                        prompt="",
                        agent_response=f"Error: {e}",
                        success=False,
                        score=0.0,
                        latency=0.0,
                        cost=0.0,
                        metadata={"error": str(e), "domain": tasks[idx].get("domain")},
                    )
                results[idx] = result
                with self._lock:
                    self._total += 1
                    self._success += 1 if result.success else 0
                    self._scores.append(float(result.score or 0.0))
                self._save_result_incremental(result)
        return [r for r in results if r is not None]

    def run(
        self,
        *,
        limit: Optional[int] = None,
        save_results: bool = True,
    ) -> List[EvaluationResult]:
        tasks = self.load_tasks(limit)
        self._total = 0
        self._success = 0
        self._scores = []

        if self.verbose:
            print(f"\nRunning Planning benchmark ({self.domain}) with {len(tasks)} tasks...")
            print(f"Agent: {self.agent.__class__.__name__}, Model: {self.agent.model}")

        if save_results:
            if self.run_dir:
                base_dir = self.run_dir
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_dir = Path("results") / "planning" / f"{self.agent.model}_{timestamp}"
            self._output_dir = base_dir / self.agent.__class__.__name__
            self._output_dir.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                print(f"Saving to: {self._output_dir}/\n")

        if self.concurrency > 1:
            results = self._run_concurrent(tasks)
        else:
            results = self._run_sequential(tasks)

        if save_results:
            self._save_summary(results)

        if self.verbose and results:
            avg = sum(self._scores) / len(self._scores)
            print(f"\nAvg composite score: {avg:.3f}, Success rate: {self._success}/{self._total}")
        return results


def _extract_constraint_keywords(query: str) -> List[str]:
    """Extract likely constraint keywords from query (cities, numbers, key terms)."""
    keywords = []
    # Cities / places (simple: capitalized words or known patterns)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query):
        keywords.append(m.group(1).strip())
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:CNY|USD|day|star)\b", query, re.IGNORECASE):
        keywords.append(m.group(0))
    return keywords[:10]  # cap
