from django.urls import include, path
from django.contrib import admin

urlpatterns = [
    path('ml/', include('ml.urls')),
    path('admin/', admin.site.urls),
]