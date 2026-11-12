"""Fictional restaurant data for Juniper and Rye, loaded into SQLite.

Everything here is invented for the demo. Allergen fields are the only source the
agent may use for allergen answers; dishes marked unverified must never get a
"safe" answer.
"""
import json
import sqlite3
from pathlib import Path

RESTAURANT = "Juniper and Rye"

# Allergen keys used across the system. "shellfish" covers crustaceans and molluscs here.
ALLERGENS = ["milk", "egg", "fish", "shellfish", "tree_nuts", "peanuts", "gluten", "soy", "sesame"]

# 0 = Monday. None = closed. Times are 24h local.
HOURS = {
    0: None,
    1: ("17:00", "22:00"),
    2: ("17:00", "22:00"),
    3: ("17:00", "22:00"),
    4: ("17:00", "23:00"),
    5: ("17:00", "23:00"),
    6: ("16:00", "21:00"),
}
LAST_BOOKING_BEFORE_CLOSE_MIN = 90
SEATING_MIN = 120
SLOT_MIN = 30
MAX_PARTY = 8

TABLES = [(f"T{i}", 2) for i in range(1, 5)] + [(f"T{i}", 4) for i in range(5, 9)] + [("T9", 6), ("T10", 6), ("T11", 8)]

# name, aliases, category, price, description, ingredients, contains, may_contain, verified, tags
MENU = [
    ("Burrata and Heirloom Tomato", ["burrata", "tomato salad"], "starter", 16,
     "Burrata with heirloom tomatoes, basil and grilled sourdough.",
     ["burrata", "heirloom tomato", "basil", "olive oil", "sourdough toast"], ["milk", "gluten"], [], True, ["vegetarian"]),
    ("Charred Carrots", ["carrots"], "starter", 14,
     "Charred carrots on tahini yoghurt with hazelnut dukkah.",
     ["carrots", "tahini", "yoghurt", "hazelnut", "sesame seeds", "coriander seed"], ["milk", "tree_nuts", "sesame"], [], True, ["vegetarian"]),
    ("Crispy Calamari", ["calamari", "squid"], "starter", 17,
     "Fried squid with lemon aioli.",
     ["squid", "wheat flour", "egg yolk", "lemon", "garlic", "rapeseed oil"], ["shellfish", "gluten", "egg"], ["fish"], True, []),
    ("Rye Sourdough and Cultured Butter", ["bread", "rye bread", "sourdough", "bread and butter"], "starter", 7,
     "House rye sourdough with cultured butter.",
     ["rye flour", "wheat flour", "salt", "cultured butter"], ["gluten", "milk"], [], True, ["vegetarian"]),
    ("Pesto Linguine", ["pesto", "linguine", "pesto pasta"], "main", 24,
     "Fresh egg linguine with basil and pine nut pesto and parmesan.",
     ["egg linguine", "basil", "pine nuts", "parmesan", "garlic", "olive oil"], ["gluten", "egg", "tree_nuts", "milk"], [], True, ["vegetarian"]),
    ("Wild Mushroom Risotto", ["risotto", "mushroom risotto"], "main", 26,
     "Arborio rice, wild mushrooms, parmesan and butter.",
     ["arborio rice", "wild mushrooms", "parmesan", "butter", "white wine", "vegetable stock"], ["milk"], [], True, ["vegetarian"]),
    ("Miso Glazed Salmon", ["salmon", "miso salmon"], "main", 31,
     "Salmon with miso glaze, bok choy and toasted sesame.",
     ["salmon", "white miso", "bok choy", "sesame seeds", "rice vinegar"], ["fish", "soy", "sesame"], [], True, []),
    ("Juniper Burger", ["burger", "cheeseburger"], "main", 22,
     "Beef patty, aged cheddar, pickles and house sauce on a brioche bun.",
     ["beef", "brioche bun", "cheddar", "pickles", "house sauce (egg, mustard)"], ["gluten", "egg", "milk"], ["sesame"], True, []),
    ("Roast Half Chicken", ["chicken", "roast chicken"], "main", 28,
     "Half chicken with salsa verde and crushed potatoes.",
     ["chicken", "capers", "anchovy", "parsley", "potatoes", "olive oil"], ["fish"], [], True, []),
    ("Peanut Noodles with Tofu", ["peanut noodles", "noodles", "tofu noodles"], "main", 21,
     "Rice noodles, crispy tofu and peanut lime sauce.",
     ["rice noodles", "tofu", "peanuts", "soy sauce", "lime", "chilli"], ["peanuts", "soy", "gluten"], [], True, ["vegan"]),
    ("Chef's Seasonal Special", ["special", "seasonal special", "squash agnolotti", "agnolotti"], "main", 27,
     "This week: squash agnolotti with sage brown butter. The recipe changes weekly.",
     ["squash", "pasta", "sage", "brown butter"], [], [], False, ["vegetarian"]),
    ("Hand-Cut Fries", ["fries", "chips"], "side", 7,
     "Hand-cut potatoes, cooked in the shared fryer.",
     ["potatoes", "rapeseed oil", "salt"], [], ["shellfish", "gluten", "fish"], True, ["vegan"]),
    ("Green Salad", ["salad", "side salad"], "side", 8,
     "Mixed leaves with shallot vinaigrette.",
     ["mixed leaves", "shallot", "dijon mustard", "red wine vinegar", "olive oil"], [], [], True, ["vegan"]),
    ("Flourless Chocolate Torte", ["chocolate torte", "chocolate cake", "torte"], "dessert", 12,
     "Dark chocolate and almond torte with crème fraîche.",
     ["dark chocolate", "eggs", "butter", "almond flour", "crème fraîche"], ["egg", "milk", "tree_nuts"], [], True, ["vegetarian"]),
    ("Olive Oil Cake", ["cake", "olive oil cake"], "dessert", 10,
     "Citrus olive oil cake.",
     ["wheat flour", "eggs", "olive oil", "orange", "sugar"], ["gluten", "egg"], [], True, ["vegetarian"]),
    ("Sorbet of the Day", ["sorbet"], "dessert", 8,
     "Flavour changes daily.",
     ["fruit", "sugar"], [], [], False, ["vegan"]),
]

# Existing bookings so that Friday 2026-11-13 around 19:30 is full for parties of 3+.
SEED_RESERVATIONS = [
    ("Okafor", 4, "2026-11-13", "19:00", "T5"), ("Lindqvist", 3, "2026-11-13", "19:00", "T6"),
    ("Moreau", 4, "2026-11-13", "19:00", "T7"), ("Tanaka", 4, "2026-11-13", "19:00", "T8"),
    ("Haddad", 5, "2026-11-13", "19:30", "T9"), ("Novak", 6, "2026-11-13", "19:30", "T10"),
    ("Reyes", 7, "2026-11-13", "18:30", "T11"), ("Byrne", 2, "2026-11-13", "19:30", "T1"),
]

SCHEMA = """
CREATE TABLE dishes (id INTEGER PRIMARY KEY, name TEXT UNIQUE, aliases TEXT, category TEXT, price REAL,
  description TEXT, ingredients TEXT, contains TEXT, may_contain TEXT, allergens_verified INTEGER, tags TEXT);
CREATE TABLE tables_ (id TEXT PRIMARY KEY, seats INTEGER);
CREATE TABLE reservations (id INTEGER PRIMARY KEY, code TEXT UNIQUE, name TEXT, party_size INTEGER,
  date TEXT, time TEXT, table_id TEXT REFERENCES tables_(id), source TEXT);
"""


def build(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Create a fresh database with the fictional data."""
    if str(path) != ":memory:":
        Path(path).unlink(missing_ok=True)
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    for name, aliases, cat, price, desc, ing, contains, may, verified, tags in MENU:
        assert set(contains) <= set(ALLERGENS) and set(may) <= set(ALLERGENS), name
        db.execute("INSERT INTO dishes (name, aliases, category, price, description, ingredients, contains, may_contain,"
                   " allergens_verified, tags) VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (name, json.dumps(aliases), cat, price, desc, json.dumps(ing), json.dumps(contains),
                    json.dumps(may), int(verified), json.dumps(tags)))
    db.executemany("INSERT INTO tables_ VALUES (?,?)", TABLES)
    for i, (name, n, d, t, table) in enumerate(SEED_RESERVATIONS, 1):
        db.execute("INSERT INTO reservations (code, name, party_size, date, time, table_id, source) VALUES (?,?,?,?,?,?,?)",
                   (f"JR-{1000 + i}", name, n, d, t, table, "seed"))
    db.commit()
    return db
