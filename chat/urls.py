# chat/urls.py
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = "chat"

urlpatterns = [
    # auth / pages
    # path("", views.login_page, name="login"),
    # path("register/", views.register_page, name="register"),
    path("", views.chat_page, name="chat"),

    # AJAX / API endpoi
    path("send_message/", views.send_message, name="send_message"),

    # Auth actions
    # path('logout/', views.logout_view, name='logout'),
]

# Serve media files during development (only when DEBUG=True)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=getattr(settings, "MEDIA_ROOT", None))
