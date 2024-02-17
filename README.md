This project is intended for graduation projects.

### TO RUN PROJECT
1. Install requirement dependencies with command: `pip install -r requirements.txt` or `sudo pip install -r requirements.txt`
2. Activate virtual eviroment: `source venv/bin/activate` #MacOS or `venv\Scripts\activate` #Window
3. Create new cert with `openssl genpkey -algorithm RSA -out key.pem` then access to `app.py` file -> change the direction of `cert.pem` and `key.pem` of your
4. RUN: `python app.py`
5. Access to your IP with https protocol, ex: `https:192.168.1.1:8000`

### Image processing mechanism
![image](https://github.com/gianam65/ai-server/assets/64204862/32893e04-c96e-40e9-be89-4b600454da8f)

### RESULT
- With this image
  
  ![image](https://github.com/gianam65/ai-server/assets/64204862/3bdbb484-0b2d-40c7-be14-b78133ed5e57)

- The result response
  answers: `
  {
    1: ['A'],
    2: ['B'],
    3: ['C'],
    4: ['D'],
    5: ['D'],
    6: ['C'],
    7: ['B'],
    8: ['A'],
    9: ['A'],
    10: ['B']
}
  `

### TODO
1. Improve the performance
2. Get student information
3. Support for grading more types of exams
