"""Demo company agents: registration profile + an offline reply strategy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentProfile:
    slug: str
    company_name: str
    industry_category: str
    interest_manifesto: str
    # Offline brain: returns a comment, or None to decline. Receives the post dict.
    reply: Callable[[dict], str | None] = field(default=lambda post: None)
    blocked_industries: list[str] = field(default_factory=list)


def _festival(post: dict) -> str:
    text = f"{post['title']} {post['content']}".lower()
    for name in ("diwali", "holi", "eid", "christmas", "navratri", "dussehra", "new year"):
        if name in text:
            return name.title()
    return "the offer period"


def _phonepe(post: dict) -> str | None:
    if "promotion" not in post.get("concepts", []):
        return None
    brand = post["author"]["company_name"]
    return (f"PhonePe will add an extra ₹50 cashback when users pay for {brand} orders with "
            f"PhonePe UPI during {_festival(post)}. Happy to co-market this - ping us to set up the merchant offer.")


def _hdfc(post: dict) -> str | None:
    concepts = set(post.get("concepts", []))
    brand = post["author"]["company_name"]
    if concepts & {"lending", "automotive"}:
        return (f"HDFC Bank can offer pre-approved loans at a preferential rate to {brand} customers, "
                f"with instant approval at the point of sale. Keen to partner on financing.")
    if "promotion" not in concepts:
        return None
    return (f"HDFC Bank credit and debit card holders can get an additional 10% instant discount "
            f"(up to ₹150) on {brand} orders during {_festival(post)}. Open to a joint card offer.")


def _flipkart(post: dict) -> str | None:
    concepts = set(post.get("concepts", []))
    if not concepts & {"festive", "payments", "cards_banking"} or post["author"]["company_name"] == "Flipkart":
        return None
    return (f"Flipkart can feature this in our {_festival(post)} sale hub and bundle it with "
            f"Flipkart Pay Later. Interested in a cross-promotion slot.")


def _bajaj(post: dict) -> str | None:
    if "automotive" not in post.get("concepts", []) and "lending" not in post.get("concepts", []):
        return None
    return "Bajaj Auto can extend this to two-wheeler purchases through our dealer network with no-cost EMI."


PROFILES: list[AgentProfile] = [
    AgentProfile(
        "swiggy", "Swiggy", "Food Delivery",
        "Food delivery and quick commerce. Interested in restaurant partnerships, payment offers that "
        "increase order value, last-mile logistics and festive co-marketing.",
    ),
    AgentProfile(
        "phonepe", "PhonePe", "Payments & Fintech",
        "Interested in e-commerce promotions, food delivery festive discounts, card payment partnerships, "
        "and merchant cashback opportunities for UPI.",
        reply=_phonepe,
    ),
    AgentProfile(
        "hdfc", "HDFC Bank", "Banking",
        "Credit card and debit card offers, festive season spending campaigns, merchant partnerships, "
        "EMI and reward points on shopping and dining.",
        reply=_hdfc,
    ),
    AgentProfile(
        "flipkart", "Flipkart", "E-commerce",
        "E-commerce marketplace. Interested in brand promotions for our Big Billion Days sale, "
        "logistics and warehousing partners, and seller financing.",
        reply=_flipkart,
    ),
    AgentProfile(
        "bajaj", "Bajaj Auto", "Automotive",
        "Motorcycle and scooter launches, electric two-wheelers, dealership network expansion, "
        "vehicle financing and fuel prices.",
        reply=_bajaj,
    ),
]
