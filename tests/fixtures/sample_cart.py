"""A small example module used in tests and demos."""

import math
from dataclasses import dataclass

try:
    import json
except ImportError:  # pragma: no cover
    json = None

MAX_ITEMS = 50
DISCOUNT_THRESHOLD = 100.0


def round_price(value: float) -> float:
    """Round a price up to the nearest cent."""
    return math.ceil(value * 100) / 100


@dataclass
class CartItem:
    name: str
    unit_price: float
    quantity: int = 1

    def total(self) -> float:
        return round_price(self.unit_price * self.quantity)


class Cart:
    def __init__(self):
        self.items: list[CartItem] = []

    def add(self, item: CartItem) -> None:
        if len(self.items) >= MAX_ITEMS:
            raise ValueError("cart is full")
        if item.quantity <= 0:
            raise ValueError("quantity must be positive")
        self.items.append(item)

    @property
    def subtotal(self) -> float:
        return sum(i.total() for i in self.items)

    def checkout_total(self) -> float:
        def apply_discount(amount: float) -> float:
            return amount * 0.9 if amount >= DISCOUNT_THRESHOLD else amount

        return round_price(apply_discount(self.subtotal))


async def export_cart(cart: Cart) -> str:
    return json.dumps([i.name for i in cart.items])
