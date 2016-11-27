from django.db import models
from datetime import datetime


class Code(models.Model):
    title = models.CharField(max_length=100)
    text = models.TextField()


    def publish(self):
        self.published_date = datetime.auto_now()
        self.save()

    def __str__(self):
        return self.title
