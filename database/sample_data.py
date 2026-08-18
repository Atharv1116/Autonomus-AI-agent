"""
Sample data generator using Faker.

Creates a realistic e-commerce database with 100k+ records across
7 tables: categories, regions, customers, employees, products, orders, sales.
All relationships are enforced via foreign keys.
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime, timedelta
from typing import Any

from faker import Faker
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

from config.logging_config import get_logger

logger = get_logger("database.sample_data")

fake = Faker()
Faker.seed(42)
random.seed(42)


def create_tables(engine: Engine) -> MetaData:
    """
    Create all sample data tables.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        MetaData with all tables defined.
    """
    metadata = MetaData()

    # --- Categories ---
    Table(
        "categories", metadata,
        Column("category_id", Integer, primary_key=True, autoincrement=True),
        Column("category_name", String(100), nullable=False),
        Column("description", Text),
        Column("created_at", DateTime, default=datetime.utcnow),
    )

    # --- Regions ---
    Table(
        "regions", metadata,
        Column("region_id", Integer, primary_key=True, autoincrement=True),
        Column("region_name", String(100), nullable=False),
        Column("country", String(100), nullable=False),
        Column("state", String(100)),
        Column("city", String(100)),
        Column("timezone", String(50)),
    )

    # --- Customers ---
    Table(
        "customers", metadata,
        Column("customer_id", Integer, primary_key=True, autoincrement=True),
        Column("first_name", String(100), nullable=False),
        Column("last_name", String(100), nullable=False),
        Column("email", String(255), unique=True),
        Column("phone", String(50)),
        Column("address", String(500)),
        Column("city", String(100)),
        Column("state", String(100)),
        Column("country", String(100)),
        Column("zip_code", String(20)),
        Column("age", Integer),
        Column("gender", String(20)),
        Column("segment", String(50)),
        Column("region_id", Integer, ForeignKey("regions.region_id")),
        Column("created_at", DateTime, default=datetime.utcnow),
    )

    # --- Employees ---
    Table(
        "employees", metadata,
        Column("employee_id", Integer, primary_key=True, autoincrement=True),
        Column("first_name", String(100), nullable=False),
        Column("last_name", String(100), nullable=False),
        Column("email", String(255)),
        Column("department", String(100)),
        Column("job_title", String(200)),
        Column("hire_date", DateTime),
        Column("salary", Float),
        Column("region_id", Integer, ForeignKey("regions.region_id")),
        Column("manager_id", Integer, ForeignKey("employees.employee_id"), nullable=True),
        Column("is_active", Integer, default=1),
    )

    # --- Products ---
    Table(
        "products", metadata,
        Column("product_id", Integer, primary_key=True, autoincrement=True),
        Column("product_name", String(200), nullable=False),
        Column("category_id", Integer, ForeignKey("categories.category_id")),
        Column("brand", String(100)),
        Column("unit_price", Float, nullable=False),
        Column("cost_price", Float),
        Column("stock_quantity", Integer, default=0),
        Column("sku", String(50), unique=True),
        Column("weight_kg", Float),
        Column("is_active", Integer, default=1),
        Column("created_at", DateTime, default=datetime.utcnow),
    )

    # --- Orders ---
    Table(
        "orders", metadata,
        Column("order_id", Integer, primary_key=True, autoincrement=True),
        Column("customer_id", Integer, ForeignKey("customers.customer_id")),
        Column("employee_id", Integer, ForeignKey("employees.employee_id")),
        Column("order_date", DateTime, nullable=False),
        Column("ship_date", DateTime),
        Column("status", String(50)),
        Column("shipping_method", String(100)),
        Column("shipping_cost", Float),
        Column("discount_percent", Float, default=0.0),
        Column("total_amount", Float),
        Column("region_id", Integer, ForeignKey("regions.region_id")),
        Column("notes", Text),
    )

    # --- Sales (line items) ---
    Table(
        "sales", metadata,
        Column("sale_id", Integer, primary_key=True, autoincrement=True),
        Column("order_id", Integer, ForeignKey("orders.order_id")),
        Column("product_id", Integer, ForeignKey("products.product_id")),
        Column("quantity", Integer, nullable=False),
        Column("unit_price", Float, nullable=False),
        Column("discount", Float, default=0.0),
        Column("total_price", Float),
        Column("sale_date", DateTime),
        Column("region_id", Integer, ForeignKey("regions.region_id")),
    )

    metadata.create_all(engine)
    logger.info("All tables created successfully")
    return metadata


def generate_sample_data(engine: Engine, scale: float = 1.0) -> None:
    """
    Populate tables with realistic sample data.

    Args:
        engine: SQLAlchemy engine.
        scale: Multiplier for record counts (1.0 = ~100k total records).
    """
    logger.info("Generating sample data (scale=%.1f)...", scale)

    # Check if data already exists
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM categories"))
            count = result.scalar()
            if count and count > 0:
                logger.info("Data already exists (%d categories). Skipping generation.", count)
                return
        except Exception:
            pass  # Table might not exist yet

    metadata = create_tables(engine)

    with engine.connect() as conn:
        # ========================================
        # 1. Categories (15 records)
        # ========================================
        categories = [
            {"category_name": "Electronics", "description": "Electronic devices and gadgets"},
            {"category_name": "Clothing", "description": "Apparel and fashion items"},
            {"category_name": "Home & Garden", "description": "Home improvement and garden supplies"},
            {"category_name": "Sports & Outdoors", "description": "Sporting goods and outdoor equipment"},
            {"category_name": "Books", "description": "Books, e-books, and publications"},
            {"category_name": "Toys & Games", "description": "Toys, games, and entertainment"},
            {"category_name": "Health & Beauty", "description": "Health, beauty, and personal care"},
            {"category_name": "Automotive", "description": "Auto parts and accessories"},
            {"category_name": "Food & Beverages", "description": "Food products and drinks"},
            {"category_name": "Office Supplies", "description": "Office and business supplies"},
            {"category_name": "Pet Supplies", "description": "Pet food and accessories"},
            {"category_name": "Jewelry", "description": "Jewelry and watches"},
            {"category_name": "Music & Instruments", "description": "Musical instruments and accessories"},
            {"category_name": "Software", "description": "Software and digital products"},
            {"category_name": "Furniture", "description": "Home and office furniture"},
        ]
        conn.execute(metadata.tables["categories"].insert(), categories)
        logger.info("Inserted %d categories", len(categories))

        # ========================================
        # 2. Regions (50 records)
        # ========================================
        regions = []
        us_states = [
            ("New York", "NY"), ("California", "CA"), ("Texas", "TX"),
            ("Florida", "FL"), ("Illinois", "IL"), ("Pennsylvania", "PA"),
            ("Ohio", "OH"), ("Georgia", "GA"), ("Michigan", "MI"),
            ("Washington", "WA"), ("Arizona", "AZ"), ("Massachusetts", "MA"),
            ("Colorado", "CO"), ("Virginia", "VA"), ("Oregon", "OR"),
        ]
        cities_by_state = {
            "NY": ["New York", "Buffalo", "Albany"],
            "CA": ["Los Angeles", "San Francisco", "San Diego", "San Jose"],
            "TX": ["Houston", "Dallas", "Austin", "San Antonio"],
            "FL": ["Miami", "Orlando", "Tampa", "Jacksonville"],
            "IL": ["Chicago", "Springfield", "Naperville"],
            "PA": ["Philadelphia", "Pittsburgh", "Harrisburg"],
            "OH": ["Columbus", "Cleveland", "Cincinnati"],
            "GA": ["Atlanta", "Savannah", "Augusta"],
            "MI": ["Detroit", "Grand Rapids", "Ann Arbor"],
            "WA": ["Seattle", "Tacoma", "Spokane"],
            "AZ": ["Phoenix", "Tucson", "Mesa"],
            "MA": ["Boston", "Cambridge", "Worcester"],
            "CO": ["Denver", "Colorado Springs", "Aurora"],
            "VA": ["Richmond", "Virginia Beach", "Norfolk"],
            "OR": ["Portland", "Salem", "Eugene"],
        }
        region_id = 1
        for state_name, state_code in us_states:
            for city in cities_by_state.get(state_code, [state_name]):
                regions.append({
                    "region_name": f"{city} Metro",
                    "country": "United States",
                    "state": state_name,
                    "city": city,
                    "timezone": "America/New_York" if state_code in ["NY", "PA", "FL", "GA", "VA", "OH", "MA"] else "America/Chicago" if state_code in ["IL", "TX"] else "America/Denver" if state_code in ["CO", "AZ"] else "America/Los_Angeles",
                })
                region_id += 1
                if len(regions) >= 50:
                    break
            if len(regions) >= 50:
                break

        conn.execute(metadata.tables["regions"].insert(), regions)
        num_regions = len(regions)
        logger.info("Inserted %d regions", num_regions)

        # ========================================
        # 3. Customers (10,000 records)
        # ========================================
        num_customers = int(10000 * scale)
        segments = ["Consumer", "Corporate", "Small Business", "Enterprise", "Government"]
        genders = ["Male", "Female", "Non-binary", "Prefer not to say"]

        customers = []
        used_emails: set[str] = set()
        for i in range(num_customers):
            email = fake.unique.email()
            while email in used_emails:
                email = fake.unique.email()
            used_emails.add(email)

            customers.append({
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": email,
                "phone": fake.phone_number()[:20],
                "address": fake.street_address(),
                "city": fake.city(),
                "state": fake.state(),
                "country": "United States",
                "zip_code": fake.zipcode(),
                "age": random.randint(18, 80),
                "gender": random.choice(genders),
                "segment": random.choice(segments),
                "region_id": random.randint(1, num_regions),
                "created_at": fake.date_time_between(start_date="-3y", end_date="now"),
            })

            # Batch insert every 2000 records
            if len(customers) >= 2000:
                conn.execute(metadata.tables["customers"].insert(), customers)
                customers = []

        if customers:
            conn.execute(metadata.tables["customers"].insert(), customers)
        logger.info("Inserted %d customers", num_customers)

        # ========================================
        # 4. Employees (200 records)
        # ========================================
        num_employees = int(200 * scale)
        departments = [
            "Sales", "Marketing", "Engineering", "Customer Support",
            "Finance", "HR", "Operations", "Legal", "Product", "Analytics",
        ]
        job_titles = {
            "Sales": ["Sales Rep", "Sales Manager", "Account Executive", "Sales Director"],
            "Marketing": ["Marketing Analyst", "Marketing Manager", "Content Strategist"],
            "Engineering": ["Software Engineer", "Data Engineer", "DevOps Engineer"],
            "Customer Support": ["Support Agent", "Support Manager", "Technical Support"],
            "Finance": ["Financial Analyst", "Controller", "Accountant"],
            "HR": ["HR Specialist", "HR Manager", "Recruiter"],
            "Operations": ["Operations Manager", "Logistics Coordinator", "Supply Chain Analyst"],
            "Legal": ["Legal Counsel", "Compliance Officer"],
            "Product": ["Product Manager", "Product Designer", "UX Researcher"],
            "Analytics": ["Data Analyst", "BI Analyst", "Data Scientist"],
        }

        employees = []
        for i in range(num_employees):
            dept = random.choice(departments)
            employees.append({
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "email": fake.company_email(),
                "department": dept,
                "job_title": random.choice(job_titles[dept]),
                "hire_date": fake.date_time_between(start_date="-10y", end_date="now"),
                "salary": round(random.uniform(40000, 180000), 2),
                "region_id": random.randint(1, num_regions),
                "manager_id": random.randint(1, max(1, i)) if i > 10 else None,
                "is_active": 1 if random.random() > 0.1 else 0,
            })
        conn.execute(metadata.tables["employees"].insert(), employees)
        logger.info("Inserted %d employees", num_employees)

        # ========================================
        # 5. Products (500 records)
        # ========================================
        num_products = int(500 * scale)
        brands = [
            "TechPro", "GreenLife", "UrbanStyle", "FitGear", "HomeComfort",
            "SmartChoice", "PureNature", "EliteEdge", "ValueMax", "ProStar",
            "NexGen", "CorePlus", "EcoSmart", "PrimeLine", "TopTier",
        ]

        products = []
        for i in range(num_products):
            unit_price = round(random.uniform(5, 2000), 2)
            cost_price = round(unit_price * random.uniform(0.3, 0.75), 2)
            products.append({
                "product_name": f"{random.choice(brands)} {fake.catch_phrase()}"[:200],
                "category_id": random.randint(1, 15),
                "brand": random.choice(brands),
                "unit_price": unit_price,
                "cost_price": cost_price,
                "stock_quantity": random.randint(0, 5000),
                "sku": f"SKU-{i+1:06d}",
                "weight_kg": round(random.uniform(0.1, 50), 2),
                "is_active": 1 if random.random() > 0.05 else 0,
                "created_at": fake.date_time_between(start_date="-2y", end_date="now"),
            })
        conn.execute(metadata.tables["products"].insert(), products)
        logger.info("Inserted %d products", num_products)

        # ========================================
        # 6. Orders (30,000 records)
        # ========================================
        num_orders = int(30000 * scale)
        statuses = ["Completed", "Shipped", "Processing", "Cancelled", "Returned", "Pending"]
        shipping_methods = ["Standard", "Express", "Overnight", "Two-Day", "Ground", "International"]

        orders = []
        for i in range(num_orders):
            order_date = fake.date_time_between(start_date="-2y", end_date="now")
            ship_date = order_date + timedelta(days=random.randint(1, 14)) if random.random() > 0.15 else None
            status = random.choice(statuses)
            if ship_date is None and status in ("Shipped", "Completed"):
                status = "Processing"

            discount = random.choice([0, 0, 0, 5, 10, 15, 20, 25])

            orders.append({
                "customer_id": random.randint(1, num_customers),
                "employee_id": random.randint(1, num_employees),
                "order_date": order_date,
                "ship_date": ship_date,
                "status": status,
                "shipping_method": random.choice(shipping_methods),
                "shipping_cost": round(random.uniform(0, 50), 2),
                "discount_percent": float(discount),
                "total_amount": 0.0,  # Will be updated after sales
                "region_id": random.randint(1, num_regions),
                "notes": fake.sentence() if random.random() > 0.8 else None,
            })

            if len(orders) >= 2000:
                conn.execute(metadata.tables["orders"].insert(), orders)
                orders = []

        if orders:
            conn.execute(metadata.tables["orders"].insert(), orders)
        logger.info("Inserted %d orders", num_orders)

        # ========================================
        # 7. Sales / Line Items (80,000 records)
        # ========================================
        num_sales = int(80000 * scale)

        sales = []
        for i in range(num_sales):
            order_id = random.randint(1, num_orders)
            product_id = random.randint(1, num_products)
            quantity = random.randint(1, 20)
            unit_price = round(random.uniform(5, 2000), 2)
            discount = round(random.uniform(0, 0.3), 2)
            total = round(quantity * unit_price * (1 - discount), 2)

            sales.append({
                "order_id": order_id,
                "product_id": product_id,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": discount,
                "total_price": total,
                "sale_date": fake.date_time_between(start_date="-2y", end_date="now"),
                "region_id": random.randint(1, num_regions),
            })

            if len(sales) >= 5000:
                conn.execute(metadata.tables["sales"].insert(), sales)
                sales = []

        if sales:
            conn.execute(metadata.tables["sales"].insert(), sales)
        logger.info("Inserted %d sales records", num_sales)

        # Commit all changes
        conn.commit()

    logger.info("Sample data generation complete!")


def seed_database(database_url: str, scale: float = 1.0) -> None:
    """
    Full database seeding entrypoint.

    Args:
        database_url: SQLAlchemy database URL.
        scale: Data scale multiplier.
    """
    engine = create_engine(database_url)
    generate_sample_data(engine, scale=scale)
    engine.dispose()


if __name__ == "__main__":
    """Run as standalone script: python -m database.sample_data"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/analyst_db")

    print(f"Seeding database: {db_url}")
    seed_database(db_url)
    print("Done!")
