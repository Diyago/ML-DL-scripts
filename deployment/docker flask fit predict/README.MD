This is baseline for model deployment via docker and flask.


1. Install
```
sudo docker-compose up
```

2. Build
```
sudo docker-compose build
```

3. Run
```
sudo docker-compose up
```

4. Get to the docker
```
sudo docker exec -it  flaskhello_webapp_1 bash
```

5. Get active dockers
```
sudo docker ps
```

6. To run python file inside the docker
```
sudo docker exec -it  flaskhello_webapp_1 python train_model.py
```

7. POST JSON data with Curl from a terminal/commandline to test spring REST
```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"flower":"1,2,3,7"}' \
  http://localhost:5000/iris_post
```

7. POST curl to upload file
```
curl -X POST -F file=@"/home/dex/Desktop/ml/flask-hello/static/setosa.jpg" http://localhost:5000/upload
```
