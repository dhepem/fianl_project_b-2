from django.shortcuts import render, redirect
from .models import Post
# Create your views here.


def index(request):
    # return HttpResponse("Hello World!")
    if request.method == "POST":
        post = Post()
        post.image = request.FILES['image']
        post.save()
        return render(request, 'index.html', {'post': post})
    else:
        post = Post()
        return render(request, 'index.html', {'post': post})
