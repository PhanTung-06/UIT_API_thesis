from locust import HttpUser, TaskSet, task

class UserBehavior(TaskSet):
    @task
    def predict(self):
        with open('IMG_2002.jpg', 'rb') as image:
            self.client.post('/predict', files={'img_file': image})
    
    @task
    def home(self):
        self.client.get('/')


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    min_wait = 500
    max_wait = 5000
