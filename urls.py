from django.urls import path
from . import views

urlpatterns = [
    path("", views.chatbot_ui),
    path("api/chat/", views.chatbot_response),
    path("api/feedback/", views.record_feedback),
    path("Checklist", views.checklist, name='Checklist'),
    # path("homepage", views.home, name='homepage'),
    path("moodtracker", views.moodtracker, name="moodtracker"),
    path("motivation", views.motivation, name="motivation"),
    path("Journal", views.Journal, name="Journal"),
    path("Resources", views.Resources, name="Resources"),
]
