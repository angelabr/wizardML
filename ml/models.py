from django.db import models
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField

class Dataset(models.Model):
    PREDICTION = 'PR'
    CLASSIFICATION = 'CL'
    NATURE_CHOICES = (
        (PREDICTION, 'Prediction'),
        (CLASSIFICATION, 'Classification'),
    )
    nature = models.CharField(
        max_length=2,
        choices=NATURE_CHOICES,
        default=CLASSIFICATION,
    )
    dataset_name = models.CharField(max_length=200)
    train_percent = models.IntegerField(default=60)
    eval_percent = models.IntegerField(default=20)
    test_percent = models.IntegerField(default=20)
    samples_number = models.IntegerField(default=0)
    labels_number = models.IntegerField(default=0) 

    def __str__(self):
        return self.dataset_name

class Label(models.Model):
    CATEGORY = 'CA'
    NUMERIC = 'NU'
    DATATYPE_CHOICES = (
    	(CATEGORY, 'Category'),
    	(NUMERIC, 'Numeric'),
    )
    INPUT = 'IN'
    OUTPUT = 'OU'
    IGNORE = 'IG'
    USE_AS_CHOICES = (
        (INPUT, 'Input'),
        (OUTPUT, 'Output'),
        (IGNORE, 'Ignore'),
    )
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    label_name = models.CharField(max_length=200)
    datatype = models.CharField(
        max_length=2,
        choices=DATATYPE_CHOICES,
        default=NUMERIC,
    )
    use_as = models.CharField(
        max_length=2,
        choices=USE_AS_CHOICES,
        default=INPUT,
    )
    samples = ArrayField(models.CharField(max_length=200), default={''})
    
    def __str__(self):
        return self.label_name 