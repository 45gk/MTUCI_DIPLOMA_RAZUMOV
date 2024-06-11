from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView
from django.contrib.auth import views as auth_views
from .views import *

urlpatterns = [
    path('', welcome_view, name='main'),

    path('registration/', registration, name='registration'),
    path('regi/', registration_view, name='registration_view'),
    path('login_view/', login_view, name='login_view'),
    path('login/', check_verify, name='login'),
    path('logout/<str:username>/', logout_func, name='logout'),

    path('user_main/<str:username>/', user_main, name='user_main'),
    path('profile/<str:username>/', update_user, name='profile'),
    path('folder_analytics/<str:username>/', folder_analytics, name='folder_analytics'),
    path('sales_dashboard/<str:username>/', sales_dashboard, name='sales_dashboard'),

    path('positions/<str:username>/', position_list, name='position_list'),
    path('products/<str:username>/', product_list, name='product_list'),
    path('customers/<str:username>/', customer_list, name='customer_list'),
    path('sales/<str:username>/', sale_list, name='sale_list'),
    path('add_file/', add_file, name='add_file'),
    path('add_folder/', add_folder, name='add_folder'),
    path('delete_file/<path:relative_path>/<str:file_name>/', delete_file, name='delete_file'),
    # path('index/<path:relative_path>/', index, name='index'),



    path('bad_auth/', TemplateView.as_view(template_name='faceapp/bad_auth.html'), name='bad_auth'),
    path('face_none/', TemplateView.as_view(template_name='faceapp/face_none.html'), name='face_none'),
    path('face_many/', TemplateView.as_view(template_name='faceapp/face_many.html'), name='face_many'),
    path('unauthorized_access/', TemplateView.as_view(template_name='faceapp/unauthorized_access.html'), name='unauthorized_access'),
    path('login_required/', TemplateView.as_view(template_name='faceapp/login_required_html.html'), name='login_required'),
    path('already_regi/', TemplateView.as_view(template_name='faceapp/already_regi.html'), name='already_regi'),


    path('user_folder/<str:username>/', index, name='user_folder'),
    path('user_folder/<str:username>/add_file/', add_file, name='add_file'),
    path('user_folder/<str:username>/add_folder/', add_folder, name='add_folder'),
    path('user_folder/<str:username>/upload_file/', upload_file, name='upload_file'),
    path('user_folder/<str:username>/delete_file/<str:file_name>/', delete_file, name='delete_file'),
    # path('accounts/login/', auth_views.LoginView.as_view(), name='login'),  # Добавить этот маршрут

]
