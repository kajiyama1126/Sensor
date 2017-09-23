import numpy as np

class Sensor(object):
    def __init__(self,n,m,p,name):
        self.position = np.array(p)
        self.n = n
        self.m = m
        self.name = name

    def distance(self,x):
        return np.linalg.norm(self.position-x,2)**2

    def send_name(self):
        return self.name



class Anchor_Sensor(Sensor):
    def __init__(self,n,m,p,name):
        super(Anchor_Sensor, self).__init__(n,m,p,name)

    def send_position(self):
        return self.position




class Agent_Sensor(Sensor):
    def __init__(self,n,m,p,name,weight):
        super(Agent_Sensor, self).__init__(n,m,p,name)
        self.weight = weight

        self.x_i = np.random.rand(self.n*self.m)
        self.x_j = np.zeros([self.n,self.n *self.m])

        # self.A = np.kron(np.array([[0,1],[1,0]]),np.identity(m))
        self.b = np.zeros(self.n * self.n)
        # self.A = np.array([0,1],[1,0])
        # one_vec = np.reshape(np.ones(self.n),(-1,1))
        # print(one_vec)
        # self.x_i = np.array([[1.1,1.2,2.1,2.2]])
        # self.g = np.kron(self.x_i,one_vec)
        # self.h = np.kron(one_vec,self.x_i)
        # self.G = self.make_G()
        # self.grad_G()
        # self.grad_f()
        # # print(self.g)
        # # print(self.G)
        # # print(self.A)
    def send_position(self):
        return self.position

    def set_b(self,distance,name):
        self.b[self.name * self.n + name] = distance

    def make_G(self):
        G = np.zeros([self.n * self.n])
        # print(G)
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.m):
                    # print(self.x_i[int(self.m*i + k)]-self.x_i[int(self.m*j + k)])
                    G[i*int(self.n) + j] += (self.x_i[int(self.m*i + k)]-self.x_i[int(self.m*j + k)])**2
        # print(G)
        return G

    def grad_G(self):
        grad_G = np.zeros([self.n * self.n,self.n * self.m])
        for i in range(self.n):
            for j in range(self.n):
                 for k in range(self.n):
                     for ell in range(self.m):
                        if i == k:
                            grad_G[int(self.n * i + j)][int(self.m * k+ ell)] = 2*(self.x_i[int(self.m*i + ell)]-self.x_i[int(self.m*j+ell)])
                        elif j == k:
                            grad_G[int(self.n * i + j)][int(self.m * k + ell)] = -2*(self.x_i[int(self.m * i + ell)] - self.x_i[int(self.m * j + ell)])
                        else:
                            grad_G[int(self.n * i + j)][int(self.m * k + ell)] = 0
                        # grad_G[self.n * i + j][self.m * k  = 2(self.x_i[2*i]-self.x_i[2*j])
        return grad_G
        print(grad_G)

    def grad_f(self):
        grad_G = self.grad_G()
        tmp =  np.dot(grad_G.T,(self.make_G()-self.b))
        return tmp
        print(tmp)

    def Cost(self):
        G = self.make_G()
        return 1/2 * np.linalg.norm(G-self.b,2)**2

    def send(self):
        return self.x_i, self.name

    def receive(self,x,name):
        self.x_j[name] = x

    def update(self,k):
        self.x_j[self.name] = self.x_i

        self.x_i = np.dot(self.weight,self.x_j) - self.s(k)*self.grad_f()

    def s(self,k):
        return 0.1/(k+10)

class Agent_Sensor_anchor(Agent_Sensor):
    def __init__(self,n,m,p,name,weight,anch_n):
        super(Agent_Sensor_anchor, self).__init__(n,m,p,name,weight)

        self.anchor_n = anch_n
        self.anchor_b = np.zeros(anch_n)
        self.anchor_posi = np.zeros([self.anchor_n *self.m])

    def get_Anchor_position(self,x,anchor_name):
        self.anchor_posi[anchor_name] = x[0]
        self.anchor_posi[anchor_name + 1] = x[1]

    def get_Anchor_distance(self,dist,anchor_name):
        self.anchor_b[anchor_name] = dist

    # def grad_anchor(self):
    #     grad = np.zeros([self.n * self.m])
    #     for i in range(self.anchor_n):
    #         for k in range(self.m):
    #             grad[int(2*self.name + k)]

    def make_H(self):
        H = np.zeros([self.anchor_n])
        # print(G)
        for i in range(self.anchor_n):
            for k in range(self.m):
                H[i] += (self.x_i[int(self.name*self.m + k)]-self.anchor_posi[int(self.m*i + k)])**2
        return H

    def grad_H(self):
        grad_H = np.zeros([self.anchor_n,self.n * self.m])
        for i in range(self.anchor_n):
             for k in range(self.n):
                 for ell in range(self.m):
                     if k == self.name:
                         grad_H[i][int(k * self.m + ell)] = 2*(self.x_i[self.name * self.m + ell]-self.anchor_posi[self.m * i+ell])
        print(grad_H)
        return grad_H
        # print(grad_G)
    def grad_f1(self):
        grad_H = self.grad_H()
        tmp = np.dot(grad_H.T,self.make_H()-self.anchor_posi)
        return tmp

    def grad(self):
        return self.grad_f()+self.grad_f1()

    def update(self, k):
        self.x_j[self.name] = self.x_i

        self.x_i = np.dot(self.weight, self.x_j) - self.s(k) * self.grad()

    # def grad(self):
    #     grad = np.zeros(self.n*self.m)
    #     for i in range(self.n * self.m):
    #         grad[i] +=
    # def grad(self):
    #     grad_g = np.zeros([self.n*self.n,self.n*self.m])
        # for i in range(self.n)
        # y = self.x-np.dot(self.A,self.x)
        # return 1/2 *np.dot((self.x-np.dot(self.A.T,self.x)),(np.dot(y,y)-self.b))
if __name__=='__main__':
    n = 4
    m = 2
    weight = np.array([[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25],[0.25,0.25,0.25,0.25]])
    test = 10000
    np.random.seed(0)
    # sensor_position = np.random.rand(n,m)
    sensor_position = np.array([[0.3,0.3],[0.7,0.7],[0.4,0.4],[0.6,0.6]])
    Anchor_position = np.array([[0,0],[1,1]])
    Sensor = []
    Anchor = []
    n_a = 2
    for i in range(n):
        Sensor.append(Agent_Sensor_anchor(n,m,sensor_position[i],i,weight[i],n_a))
    for i in range(n_a):
        Anchor.append(Anchor_Sensor(n,m,Anchor_position[i],i))

    for i in range(n):
        for j in range(n):
            tmp = Sensor[i].send_position()
            dist = Sensor[j].distance(tmp)
            name = Sensor[j].send_name()
            Sensor[i].set_b(dist,name)
    for i in range(n):
        for j in range(n_a):
            tmp = Sensor[i].send_position()
            dist = Anchor[j].distance(tmp)
            name = Anchor[j].send_name()
            posi = Anchor[j].send_position()
            Sensor[i].get_Anchor_position(posi,name)
            Sensor[i].get_Anchor_distance(dist,name)


    for k in range(test):
        for i in range(n):
            for j in range(n):
                x ,name = Sensor[i].send()
                Sensor[j].receive(x,name)

        for i in range(n):
            Sensor[i].update(k)

        print(Sensor[0].x_i)