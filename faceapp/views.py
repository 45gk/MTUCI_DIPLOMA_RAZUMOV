import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings
import os
import base64
from datetime import timedelta
import numpy as np
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404
from django.db.models import Sum
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework.response import Response
import os
from collections import Counter
from .models import User, Sale, Product, Customer, Position
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import cv2
# from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from django.shortcuts import redirect
import io
from django.urls import reverse
from collections import defaultdict
from django.http import HttpResponse

cur_user = ''

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)  # initializing mtcnn for face detection
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()  # initializing resnet for face


def login(request, user):
    global cur_user
    cur_user = user.username


def logout(request, user):
    global cur_user
    cur_user = ''


def check_auth(request, user):
    global cur_user
    return cur_user == user



'''Обновление сети'''


def init_net():
    dataset = datasets.ImageFolder('faces')  # photos folder path
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names


    def collate_fn(x):
        return x[0]


    loader = DataLoader(dataset, collate_fn=collate_fn)

    face_list = []
    name_list = []
    embedding_list = []

    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob > 0.90:
            emb = facenet_model(face.unsqueeze(0))
            embedding_list.append(emb.detach())
            name_list.append(idx_to_class[idx])

    data = [embedding_list, name_list]
    torch.save(data, 'data/data1.pt')  # saving data.pt file


'''Метод обработки изображения с помощью сети'''


def face_match(img_path, data_path):

    img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True)

    #проверка на отсутствие лица
    if face is None:
        return ["No face"]

    emb = facenet_model(face.unsqueeze(0)).detach()

    saved_data = torch.load(data_path)
    embedding_list = saved_data[0]
    name_list = saved_data[1]
    dist_list = []

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    if min(dist_list) < 0.88:
        return [name_list[idx_min], min(dist_list)]
    else:
        return ['bad_auth']



@api_view(['POST'])
def registration(request):
        data_path = 'data/data1.pt'
        print(request.POST)
        required_fields = ['login', 'name', 'biograghy', 'my_skills']
        missing_fields = [field for field in required_fields if field not in request.POST]

        if missing_fields:
            return JsonResponse({'error': f'Missing fields: {", ".join(missing_fields)}'}, status=400)

        if 'photo' not in request.POST:
             return JsonResponse({'error': 'Missing photo'}, status=400)

        if User.objects.count() >= 10:
            return redirect('welcome/')

        login_cur = request.POST['login']
        name = request.POST['name']
        biograghy = request.POST['biograghy']
        my_skills = request.POST['my_skills']
        photo_data = request.FILES['photo']

        image = Image.open(photo_data)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_name = f'my_face_image.jpeg'
        # image_path = default_storage.save(image_name, ContentFile(image.read()))
        image.save(image_name)
        image_path = 'my_face_image.jpeg'


        face_match_res = face_match(image_path, data_path)

        # проверка на уже регистрацию
        if face_match_res[0] != 'bad_auth':
            if face_match_res[0] == "No face":
                return JsonResponse({'url': f'face_none/'})
            else:
                return JsonResponse({'url': f'face_none/'})

        # Создание путей к папкам
        path_to_folder = os.path.join("users_directory", login_cur)
        path_to_faces = os.path.join("faces", login_cur)

        # Создание папок, если они еще не существуют
        os.makedirs(path_to_folder, exist_ok=True)
        os.makedirs(path_to_faces, exist_ok=True)


        image_name = f'faces/{login_cur}/{login_cur}.jpeg'


        # file_bytes = np.asarray(photo_data, dtype=np.uint8)
        # print(file_bytes)
        img = cv2.imread(image_path)

        # # Применение аугментаций
        augmented_images = []
        augmented_images.append(('original.jpeg', img))

        # Поворот изображения
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        augmented_images.append(('rotated.jpeg', rotated_img))

        # Изменение яркости и контраста
        brightness_img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        augmented_images.append(('brightness.jpeg', brightness_img))

        # Отражение изображения
        flipped_img = cv2.flip(img, 1)
        augmented_images.append(('flipped.jpeg', flipped_img))

        # Сохранение изображений
        for name, aug_img in augmented_images:
            try:
                _, buffer = cv2.imencode('.jpeg', aug_img)
                file_content = ContentFile(buffer.tobytes())
                default_storage.save(f'faces/{login_cur}/', file_content)
            except BaseException:
                pass
        image_path = default_storage.save(image_name, photo_data)
        init_net()

        User.objects.create(
            username=login_cur,
            password='123',
            name_user=name,
            # feature_vector='LOL',
            path_to_folder=path_to_folder,
            biograghy=biograghy,
            my_skills=my_skills,
            path_to_faces=path_to_faces,
        )

        user = User.objects.get(username=login_cur)
        login(request, user)
        print('Regi finished')
        return JsonResponse({'url': f'user_main/{user.username}/'})


'''Метод, подтверждающий авторизацию пользователя'''


@api_view(['POST'])
def check_verify(request):
    data_path = 'data/data1.pt'
    image = request.FILES['image']
    image = Image.open(image)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image_name = f'my_face_image.jpeg'
    # image_path = default_storage.save(image_name, ContentFile(image.read()))
    image.save(image_name)
    image_path = 'my_face_image.jpeg'

    login_res = face_match(image_path, data_path)
    if login_res[0] != 'No face' and login_res[0] != 'bad_auth':
        login_cur = login_res[0]
    elif login_res[0] != 'No face':
        login_cur = None
    else:
        login_cur = 'bad_auth'

    if login_cur is not None and login_cur != 'bad_auth':
        print(login_cur)
        user = User.objects.get(username=login_cur)
        login(request, user)
        # url = reverse()
        return JsonResponse({'url': f'user_main/{user.username}/'})
    elif login_cur == 'bad_auth':
        print("Неудачная авторизация")
        bad_auth_url = reverse('bad_auth')
        return JsonResponse({'url': bad_auth_url})

    else:
        print("Нет лица")
        bad_auth_url = reverse('face_none')
        return JsonResponse({'url': bad_auth_url})


def login_view(request):
    return render(request, 'faceapp/login.html')


@csrf_exempt
def registration_view(request):
    return render(request, 'faceapp/registration.html')


'''Метод отображения главной страницы авторизованного пользователя'''


# @login_required
@api_view(['GET'])
def user_main(request, username):
    print(username)
    print(cur_user)

    if username == cur_user:
        user = User.objects.get(username=username)
        # print(user)
        total_files = 0
        file_types_count = defaultdict(int)

        # Проходим по всем файлам в папке пользователя
        for root, dirs, files in os.walk(user.path_to_folder):
            for file in files:
                total_files += 1
                file_extension = os.path.splitext(file)[1].lower()
                file_types_count[file_extension] += 1
        context = {
            'username': user.username,
            'total_files': total_files,
            'name': user.name_user,
            'biograghy': user.biograghy,
            'my_skills': user.my_skills
        }
        return render(request, 'faceapp/user_info.html', context=context)

    return redirect('unauthorized_access')


'''Метод, отображающий содержимое каталога'''



@api_view(['GET'])
def user_folder(request, username):
    if check_auth(request, username):
        user = User.objects.get(username=username)
        folder_path = user.path_to_folder
        files = os.listdir(folder_path)
        return render(request, 'faceapp/folder.html', {'user': user, 'files': files})
    else:
        return redirect('unauthorized_access')


'''Метод для аналитики содержимого каталога'''


@api_view(['GET'])
def folder_analytics(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    # Получаем пользователя по имени пользователя
    user = get_object_or_404(User, username=username)

    # Получаем путь к папке пользователя
    path_to_folder = user.path_to_folder

    # Инициализируем счетчики
    total_files = 0
    file_types_count = defaultdict(int)

    # Проходим по всем файлам в папке пользователя
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            total_files += 1
            file_extension = os.path.splitext(file)[1].lower()
            file_types_count[file_extension] += 1

    # Формируем данные для шаблона
    context = {
        'username': username,
        'total_files': total_files,
        'file_types_count': dict(file_types_count)
    }

    # Рендерим HTML-шаблон с данными
    return render(request, 'faceapp/folder_analytics.html', context)


def file_type_analytics(directory_hash):
    directory_path = os.path.join('user_directories', str(directory_hash))
    file_types = [os.path.splitext(file)[1] for file in os.listdir(directory_path)]
    return dict(Counter(file_types))


@login_required
@api_view(['GET'])
def count_files(directory_hash):
    directory_path = os.path.join('user_directories', str(directory_hash))
    return len(os.listdir(directory_path))


@api_view(['GET'])
def position_list(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    positions = Position.objects.all()
    return render(request, 'faceapp/position_list.html', {'positions': positions})


@api_view(['GET'])
def product_list(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    products = Product.objects.all()
    return render(request, 'faceapp/product_list.html', {'products': products})


@api_view(['GET'])
def customer_list(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    customers = Customer.objects.all()
    return render(request, 'faceapp/customer_list.html', {'customers': customers})


@api_view(['GET'])
def sale_list(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    sales = Sale.objects.all()
    return render(request, 'faceapp/sales_list.html', {'sales': sales})



'''Создание каталога'''


@login_required
@api_view(['POST'])
def create_directory(directory_hash):
    directory_path = os.path.join('user_directories', str(directory_hash))
    os.makedirs(directory_path, exist_ok=True)


# @login_required
@api_view(['GET'])
def profile(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    user = User.objects.get(username=username)
    return render(request, 'profile.html', {'user': user})


@api_view(['GET'])
def logout_func(request, username):
    user = User.objects.get(username=username)
    logout(request, user)
    return redirect('login_view')

@api_view(['GET'])
def welcome_view(request):
    return render(request, 'faceapp/welcome.html')


@api_view(['GET'])
def catalogues_view(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    return render(request, 'faceapp/catalogues.html')


@api_view(['GET'])
def analytics_view(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    return render(request, 'faceapp/analytics.html')


def update_user(request, username):
    if cur_user == '':
        return redirect('unauthorized_access')
    if username != cur_user:
        return redirect('login_required')

    user = get_object_or_404(User, username=username)
    if request.method == 'POST':
        # user.username = request.POST['login']
        # user.name_user = request.POST['name']
        user.biograghy = request.POST['biograghy']
        user.my_skills = request.POST['my_skills']
        user.save()
        return redirect('user_main', username=user.username)
    return render(request, 'faceapp/profile.html', {'user': user})


@api_view(['GET'])
def sales_dashboard(request, username):
    if not check_auth(request, username):
        return redirect('unauthorized_access')
    # Получение общих продаж
    total_sales = Sale.objects.aggregate(Sum('amount'))['amount__sum']

    # Получение продаж по категориям продуктов
    product_sales = Sale.objects.values('product__name').annotate(total=Sum('amount'))

    # Получение продаж по позициям
    position_sales = Sale.objects.values('position__title').annotate(total=Sum('amount'))

    # Получение продаж по клиентам
    customer_sales = Sale.objects.values('customer__first_name', 'customer__last_name').annotate(total=Sum('amount'))

    # Получение продаж за последний месяц
    one_month_ago = timezone.now() - timedelta(days=30)
    sales_last_month = Sale.objects.filter(sale_date__gte=one_month_ago)

    dates = [sale.sale_date for sale in sales_last_month]
    sales = [sale.amount for sale in sales_last_month]

    # Создание графиков
    fig, ax = plt.subplots()
    product_names = [sale['product__name'] for sale in product_sales]
    product_totals = [sale['total'] for sale in product_sales]
    ax.bar(product_names, product_totals)
    ax.set_title('Sales by Product')
    ax.set_xlabel('Product')
    ax.set_ylabel('Total Sales')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    product_sales_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.clf()

    fig, ax = plt.subplots()
    position_titles = [sale['position__title'] for sale in position_sales]
    position_totals = [sale['total'] for sale in position_sales]
    ax.bar(position_titles, position_totals)
    ax.set_title('Sales by Position')
    ax.set_xlabel('Position')
    ax.set_ylabel('Total Sales')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    position_sales_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.clf()

    fig, ax = plt.subplots()
    customer_names = [f"{sale['customer__first_name']} {sale['customer__last_name']}" for sale in customer_sales]
    customer_totals = [sale['total'] for sale in customer_sales]
    ax.bar(customer_names, customer_totals)
    ax.set_title('Sales by Customer')
    ax.set_xlabel('Customer')
    ax.set_ylabel('Total Sales')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    customer_sales_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.clf()

    return render(request, 'faceapp/sales_dashboard.html', {
        'total_sales': total_sales,
        'product_sales_image': product_sales_image,
        'position_sales_image': position_sales_image,
        'customer_sales_image': customer_sales_image,
        'dates': dates,
        'sales': sales,
    })


BASE_DIR = os.path.join(settings.BASE_DIR, 'users_directory')


def get_user_folder_path(username):
    return os.path.join(BASE_DIR, username)


def index(request, username):
    folder_path = get_user_folder_path(username)
    if not os.path.exists(folder_path):
        return HttpResponse("Folder not found", status=404)

    files_and_folders = os.listdir(folder_path)
    items = [{'name': item, 'is_dir': os.path.isdir(os.path.join(folder_path, item))} for item in files_and_folders]

    return render(request, 'faceapp/folder.html', {'items': items, 'username': username})


def add_file(request, username):
    file_name = request.POST.get('name')
    if file_name:
        open(os.path.join(get_user_folder_path(username), file_name), 'w').close()
    return redirect(f'/user_folder/{username}/')


def delete_file(request, username, file_name):
    os.remove(os.path.join(get_user_folder_path(username), file_name))
    return redirect(f'/user_folder/{username}/')


def add_folder(request, username):
    folder_name = request.POST.get('name')
    if folder_name:
        os.makedirs(os.path.join(get_user_folder_path(username), folder_name))
    return redirect(f'/user_folder/{username}/')


def upload_file(request, username):
    folder_path = get_user_folder_path(username)
    if 'file' in request.FILES:
        file = request.FILES['file']
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
    return redirect(f'/user_folder/{username}/')
