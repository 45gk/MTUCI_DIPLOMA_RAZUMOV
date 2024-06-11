from django.contrib import admin
from .models import Position, Product, Customer, Sale, User
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User


admin.site.register(User)
admin.site.register(Position)
admin.site.register(Product)
admin.site.register(Customer)
admin.site.register(Sale)
