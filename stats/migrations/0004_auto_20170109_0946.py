# -*- coding: utf-8 -*-
# Generated by Django 1.9.6 on 2017-01-09 09:46
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stats', '0003_remove_code_input2'),
    ]

    operations = [
        migrations.AlterField(
            model_name='code',
            name='title',
            field=models.CharField(max_length=300),
        ),
    ]
