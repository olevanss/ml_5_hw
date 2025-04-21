import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys

# Добавляем текущую директорию в путь для импорта
sys.path.append('.')

# Импортируем класс модели
from model import Generator, CycleGAN

st.title("День-Ночь Город")

# Функция загрузки модели (без кеширования и обработки ошибок)
def load_model():
    model_path = "daynight_cityview_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем весовой файл
    checkpoint = torch.load(model_path, map_location=device)
    
    # Создаем модель
    cyclegan = CycleGAN(generator_channels=64, discriminator_channels=64, use_dropout=False)
    
    # Загружаем веса
    if 'gen_a2b' in checkpoint and 'gen_b2a' in checkpoint:
        cyclegan.gen_a2b.load_state_dict(checkpoint['gen_a2b'])
        cyclegan.gen_b2a.load_state_dict(checkpoint['gen_b2a'])
    elif 'state_dict' in checkpoint:
        cyclegan.load_state_dict(checkpoint['state_dict'])
    else:
        gen_a2b_weights = {k.replace('gen_a2b.', ''): v for k, v in checkpoint.items() 
                          if k.startswith('gen_a2b.')}
        gen_b2a_weights = {k.replace('gen_b2a.', ''): v for k, v in checkpoint.items() 
                          if k.startswith('gen_b2a.')}
        
        cyclegan.gen_a2b.load_state_dict(gen_a2b_weights)
        cyclegan.gen_b2a.load_state_dict(gen_b2a_weights)
    
    cyclegan.eval()
    return cyclegan

# Параметры нормализации для дневных и ночных изображений
DAY_MEAN = [0.40439875, 0.43620292, 0.4515344]
DAY_STD = [0.27975968, 0.27630162, 0.30779361]
NIGHT_MEAN = [0.19426834, 0.15749247, 0.13275868]
NIGHT_STD = [0.23491469, 0.19607065, 0.17848439]

# Функция предобработки изображения с учетом типа
def preprocess_image(image, is_day=True):
    # Выбираем соответствующие параметры нормализации
    if is_day:
        mean = DAY_MEAN
        std = DAY_STD
    else:
        mean = NIGHT_MEAN
        std = NIGHT_STD
        
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0)

# Загрузка модели
model = load_model()

uploaded_file = st.file_uploader("Загрузи фото ГОРОДА:", type=["jpg", "jpeg", "png"])

# Выбор направления преобразования
transformation_direction = st.radio(
    "Выбери режим:",
    ["День → Ночь", "Ночь → День"]
)

if uploaded_file is not None:
    # Отображение загруженного изображения
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Исходное изображение")
    
    # Кнопка для запуска преобразования
    if st.button("Тык!"):
        # Определяем тип изображения (день или ночь) на основе выбранного преобразования
        is_day_image = transformation_direction == "День → Ночь"
        
        # Предобработка изображения с учетом типа
        input_tensor = preprocess_image(image, is_day=is_day_image)
        
        # Выбор соответствующего генератора
        if transformation_direction == "День → Ночь":
            generator = model.gen_a2b
        else:
            generator = model.gen_b2a
        
        # Применяем выбранный генератор
        with torch.no_grad():
            output = generator(input_tensor)
        
        # Обработка результата и отображение изображения
        output_img = output.squeeze(0).cpu().detach()
        
        # Преобразуем результат обратно в изображение
        # Используем обратную нормализацию в зависимости от типа результата
        if transformation_direction == "День → Ночь":
            # Результат - ночное изображение
            mean = NIGHT_MEAN
            std = NIGHT_STD
        else:
            # Результат - дневное изображение
            mean = DAY_MEAN
            std = DAY_STD
            
        # Ручная денормализация
        for t, m, s in zip(output_img, mean, std):
            t.mul_(s).add_(m)
            
        output_img = transforms.ToPILImage()(output_img.clamp(0, 1))
        st.image(output_img, caption="Результат")
        