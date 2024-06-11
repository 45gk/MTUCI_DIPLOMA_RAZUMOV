import random
from django.core.management.base import BaseCommand
from ...models import Position, Product, Customer, Sale
from faker import Faker
from decimal import Decimal

class Command(BaseCommand):
    help = 'Populate the database with sample data'

    def handle(self, *args, **kwargs):
        fake = Faker()

        # Create positions
        positions = []
        for _ in range(5):
            position = Position(
                title=fake.job(),
                description=fake.text()
            )
            position.save()
            positions.append(position)

        # Create products
        products = []
        for _ in range(10):
            product = Product(
                name=fake.word(),
                description=fake.text(),
                price=Decimal(random.uniform(10.0, 100.0)),
                stock=random.randint(1, 100)
            )
            product.save()
            products.append(product)

        # Create customers
        customers = []
        for _ in range(10):
            customer = Customer(
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                email=fake.email(),
                phone=fake.phone_number(),
                address=fake.address()
            )
            customer.save()
            customers.append(customer)

        # Create sales
        for _ in range(50):
            sale = Sale(
                customer=random.choice(customers),
                position=random.choice(positions),
                product=random.choice(products),
                quantity=random.randint(1, 10),
                total_price=Decimal(random.uniform(10.0, 1000.0)),
                amount=Decimal(random.uniform(10.0, 1000.0))
            )
            sale.save()

        self.stdout.write(self.style.SUCCESS('Successfully populated the database'))
