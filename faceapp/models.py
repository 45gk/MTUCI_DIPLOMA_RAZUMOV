# models.py
from django.contrib.auth.models import AbstractUser
import uuid
# from django_cryptography.fields import encrypt
from django.db import models


class User(AbstractUser):
    name_user = models.CharField(max_length=100)
    path_to_folder = models.CharField(max_length=2000) # , unique=True
    biograghy = models.CharField(max_length=255)
    my_skills = models.CharField(max_length=255)
    path_to_faces = models.CharField(max_length=2000) #, unique=True

    # USERNAME_FIELD = "login"

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='faceapp_user_set',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_query_name='user',
        verbose_name='groups'
    )

    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='faceapp_user_set',
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='user',
        verbose_name='user permissions'
    )


class Position(models.Model):
    title = models.CharField(max_length=255, unique=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.title


class Product(models.Model):
    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(null=True, blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField()

    def __str__(self):
        return self.name


class Customer(models.Model):
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, null=True, blank=True)
    address = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class Sale(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    position = models.ForeignKey(Position, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    sale_date = models.DateTimeField(auto_now_add=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"Sale {self.id} by {self.customer}"
