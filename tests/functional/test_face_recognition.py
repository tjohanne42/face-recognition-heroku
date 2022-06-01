from tests.conftest import client


def test_get_status_code(client):
    response = client.get('/face-recognition')
    assert response.status_code == 200


def test_post_status_code(client):
    image_path = "unknown/macron_obama.jpg"
    file_data = open(image_path, "rb")
    data = {"pic": (file_data, image_path)}
    response = client.post('/face-recognition', data=data, content_type="multipart/form-data")
    assert response.status_code == 200

