
import random
import requests
from PIL import Image
from django.shortcuts import render
from django.http import HttpResponseRedirect
from datetime import datetime, date, time
from .models import Imager
from .forms import ImageForm

def index(request):
    form = ImageForm()
    images = Imager.objects.all()
    all_images = []
    img = r'/media/images/image_14_0a8GRPH.jpg'
    print(img)
    for image in images:
        print(img)
        city_info = {
            'geopos': image.coord,
            'time': datetime.today(),
            'image': image.image,
        }
        all_images.append(city_info)
    # city_info = {
    #     'geopos': "56.5245245, 43.5353452",
    #     'time': datetime.today(),
    #     'image': img,
    # }
    # all_images.append(city_info)
    #
    context = {'all_info': all_images, 'form': form}

    # form.data['geopos'] = "56.5245245, 43.5353452"
    # form.data['image'] = img

    # if(request.method == 'POST'):
    #     form = ImageForm(request.POST)
    #     for i in range(len(all_images)):
    #         if form.data['image'] == all_images[i]['image']:
    #             break
    #         if form.data['image'] != all_images[i]['image'] and i == (len(all_images)-1):
    #             form.save()


    return render(request,'weather/index.html',context)

def about(request):
    return render(request, "weather/about.html")
# Create your views here.

def delete(request, Image):
    imager = Imager.objects.get(image=Image)
    imager.delete()
    return HttpResponseRedirect("/")