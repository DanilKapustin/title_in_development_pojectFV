from django.db import models

class Imager(models.Model):
    coord = models.CharField(max_length=200)
    image = models.ImageField(upload_to='./images')

    def __str__(self):
        return self.coord
