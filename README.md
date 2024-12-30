Capstone 1: ML Zoomcamp

curl -L -o $(pwd)/grape-quality.zip  https://www.kaggle.com/api/v1/datasets/download/mrmars1010/grape-quality

unzip grape-quality.zip 


The dataset provides detailed information about individual grape samples, including their unique identifier, variety, and geographic origin. It also quantifies quality attributes such as score, category, sugar content, and acidity. Physical characteristics like cluster weight and berry size are recorded, along with environmental factors like harvest date, sun exposure, soil moisture, and rainfall.

The provided data includes information about various grape samples, covering aspects like:

Identification: Sample ID and variety
Geographic Origin: Region
Quality Metrics: Quality score, category, sugar content, acidity
Physical Characteristics: Cluster weight, berry size
Environmental Factors: Harvest date, sun exposure, soil moisture, rainfall


pipenv install pandas==2.2.3 scikit-learn==1.6.0 flask gunicorn


```
docker build -t grape_quality_prediction .
docker run -it --rm -p 8787:8787 grape_quality_prediction
```