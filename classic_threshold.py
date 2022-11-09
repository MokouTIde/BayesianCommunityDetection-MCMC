import numpy as np
import copy
import pickle
import os
import matplotlib.pyplot as plt
import random
import networkx as nx
import datetime

starttime = datetime.datetime.now()


diffp_logpost = []
diffp_initR_repeat = []
diffp_alginitR = []
diffp_finalR = []

p_range = [round(i,4) for i in np.linspace(0.30, 0.40, 51)]
for p in p_range:

    '''
    n * (np.sqrt(p)-np.sqrt(q))**2 / np.log(n) = 1.6
    n * q / np.log(n) = 32 
    '''
    
    ## 初始化
    seed = 19980210
    size = [300,300,300]
    # p = 0.60            # 社团内部连接概率
    #p = 0.4
    q = 0.30            # 社团之间连接概率
    niter = 10
    vsteps = 100
    trial = 1
    beta1 = 1.0
    beta2 = 1.0
    xi = 1.0
    # repeat = 1        改到主程序中定义
    knowp = True
    
    R = np.diag(size)
    K = R.shape[0]
    n = np.sum(R)
    c = np.repeat(np.arange(K), np.sum(R, axis=0))
    
    ## 连接概率矩阵
    Pstar = q * np.ones([K, K]) + np.diag((p - q) * np.ones(K))
    
    
    ## 将组标签扩展，得到节点数*组数的矩阵，每行代表一个节点，第几个位置为1即分为第几组
    ## 返回结果：每个节点都是一个数组，属于的组为1，其余为0
    def Zform(e, K):
        n = len(e)  #e是节点的分组列表community assgnment，n即为网络大小，共n个节点
        Ze = np.zeros((n, K))  #生成n*K的0矩阵（初始矩阵）
        Ze[np.arange(n), e] = 1  #对每个节点（每行）打标签
        return Ze
    
    ## 根据连边概率生成邻接矩阵A
    def genA():
        Zc = Zform(c, K)  #输入c和组数K，得到扩展的分组矩阵
        Omega = np.matmul(np.matmul(Zc, Pstar), np.transpose(Zc))  #Pstar为转移矩阵（p和q未知的情况下，非平衡sbm）
        A = np.random.binomial(1, Omega)  #二项分布中采样1个数，Omega为取“正”的概率
        tmp = np.triu(A) - np.diag(np.diag(A))  #A的右上三角阵减去A的特征值
        A = tmp + np.transpose(tmp)
        return A
    
    A = genA()
    
    
    ## 更新组标签
    ## 任取一个点，将其移动到别的组，返回值为各个节点的组序号（更新后），被更新的节点标签，被更新的节点原始组，被更新的节点新组
    def updateK(e, K):
        newe = copy.deepcopy(e)  #确保多维列表复制成功
        n = len(newe)  #newe的长度
        setfull = list(range(K))  #制作一个组数K的list，setfull包含所有可能的组标签
        ind = np.random.randint(n)  #在newe中抽取一个数
        b = newe[ind]  #找到随机数在newe中的位置
        
        del setfull[newe[ind]]  #删除setfull中相应位置
        b_prime = np.random.choice(setfull)  #b'，从组列表中抽取一个其他的编号
        
        
        newe[ind] = b_prime  #更新被抽取点的组标签
        return newe, ind, b, b_prime  #返回
    
    '''
    此部分为平均场和二段跳算法使用
    def trace_L(e,b):
        _que=[] # 待抽取列表，按频数保存的相邻的组
        for _i,_b in enumerate(e):
            if _b == b: # 如果是要查的组
                for _j in range(len(A[_i])):
                    if A[_i][_j]: # 如果相邻
                        _que.append(e[_j]) #记录相邻点的组标签
        return np.random.choice(_que) #从该组相邻的组中抽取一个组
    
    def updateK_L(e,K, matO, L=1):
        newe = copy.deepcopy(e)  #确保多维列表复制成功
        n = len(newe)  #newe的长度
        
        ind = np.random.randint(n)  #在newe中抽取一个数
        b = newe[ind]  # 保存原本的组标签
        
        ## 看周围点的组 并按比例随机抽一个 组
        _b = []  # 保存周围点的组
        for i in range(len(A[ind])):
            if A[ind][i]: # 如果相邻
                _b.append(newe[i])
        b_prime = np.random.choice(_b) #我的回合 抽！
        ## 跳L次组
        for _ in range(L):
            b_prime = random.choices(range(K), weights=matO[b_prime], k=1)[0] #从该组相邻的组中抽取一个组
        
        newe[ind] = b_prime  #更新被抽取点的组标签
        return newe, ind, b, b_prime  #返回
    '''
    
    ## 更新移动后的扩展矩阵
    def updateZe(Ze, ind, b, b_prime):
        newZe = copy.deepcopy(Ze)
        newZe[ind, b] = Ze[ind, b] - 1
        newZe[ind, b_prime] = Ze[ind, b_prime] + 1
        return newZe
    
    ## 误差矩阵（混淆矩阵），Ze是扰乱后的扩展组标签，Zc是原本的扩展组标签，得到的矩阵为扰动前后判断矩阵
    def Rdiff(e, c):
        K = np.max(c) + 1
        Ze = Zform(e, K)  #扰乱后的扩展组标签
        Zc = Zform(c, K)  #原本的扩展组标签
        R = np.matmul(np.transpose(Ze), Zc).astype(int)  #原本1，扰动后1；原本2，扰动后1；原本1，扰动后2；原本2，扰动后2
        return R
    
    ## 根据节点移动情况更新网络连边情况
    def updateMat(matO, matN, ind, b, b_prime, Ze):
        A_ind = A[ind, ].reshape((-1, 1))                    #被选取点的连边情况
        tmp = (A_ind * Ze).sum(axis=0)  # dim: k             #和每块的连边数量
        new_matO = matO.copy()
        new_matO[b, :] = new_matO[b, :] - tmp                #第b个点（移动点）所在的组减去第b个点的连边情况
        new_matO[b_prime, :] = new_matO[b_prime, :] + tmp    #移动到组b_prime后的连边情况
        new_matO[:, b] = new_matO[:, b] - tmp                #对称更新
        new_matO[:, b_prime] = new_matO[:, b_prime] + tmp    #对称更新
    
        new_matN = matN.copy()
        nZe = Ze.sum(axis=0)
        new_matN[b, :] = new_matN[b, :] - nZe
        new_matN[b_prime, :] = new_matN[b_prime, :] + nZe
        new_matN[:, b] = new_matN[:, b] - nZe
        new_matN[:, b_prime] = new_matN[:, b_prime] + nZe
        new_matN[b, b] = new_matN[b, b] + 1 + 1
        # new_matN[b_prime, b_prime] = new_matN[b_prime, b_prime] + 1 - 1
        new_matN[b_prime, b] = new_matN[b_prime, b] - 1
        new_matN[b, b_prime] = new_matN[b, b_prime] - 1
        return new_matO, new_matN
    
    ## 后验分布计算方法
    def loggam_apprx(x):
        tmp = (x + 1 / 2) * np.log(x) - x + 1 / 2 * np.log(2 * np.pi)
        return tmp
    
    ## 对数后验分布的快速计算，输入为连边数，可能出现的最大连边数
    def logpostpdf_fast(matO, matN):
        matY = matN - matO
        p = np.sum(
            loggam_apprx(matO + beta1 - 1) +
            loggam_apprx(matY + beta2 - 1) -
            loggam_apprx(matN + beta1 + beta2 - 2)
        ) / 2
        return p
    
    def RIindex(e0, true_label):
        a = b = c = d = 0
        for i in range(len(e0)):
            for j in range(i, len(e0)):
                if e0[i] == e0[j]:
                    if true_label[i] == true_label[j]:
                        a += 1
                    else:
                        c += 1
                else:
                    if true_label[i] == true_label[j]:
                        d += 1
                    else:
                        b += 1
        RI = (a + b) / (a + b + c + d)
        return RI
    
    
    ## 社团识别算法主程序
    def classic_MHsampler(e0=None, niter=None, knowp=False, vsteps=100):
        '''
        Args:
            e0: init
            niter: total # of iterations
            K: # of class
            beta1, beta2: hyper-param
            A: adjacency matrix
            xi: temperature
            c: true label assignment
        '''
    
    
        # 打乱
        curr_e = copy.deepcopy(e0)  #深复制一个打乱后的标签
        curr_Ze = Zform(curr_e, K)  #创建打乱后的扩展组标签
        curr_matO = np.matmul(
            np.matmul(np.transpose(curr_Ze), A), curr_Ze)
        curr_matN = np.matmul(
            np.matmul(
                np.transpose(curr_Ze), np.ones([n, n])), curr_Ze
        ) - np.diag(np.sum(curr_Ze, axis=0))
    
        # 储存当前（打乱后的）后验分布
        logPost = []
        logPost.append(logpostpdf_fast(curr_matO, curr_matN))
        # logPost.append(self.logpostpdf(e0, knowp))
        n_mis = []
    
        # 开始循环
        for i in np.arange(1, niter):
            new_e, ind, b, b_prime = updateK(curr_e, K)  #任取一个点，将其移动到别的组，返回新的标签，被移动点，移动前后的组号
            # new_e, ind, b, b_prime = updateK_L(curr_e, 0)
            new_Ze = updateZe(curr_Ze, ind, b, b_prime)  #更新扩展组标签Ze
            new_matO, new_matN = updateMat(              #更新网络的连边数量
                 curr_matO, curr_matN, ind, b, b_prime, curr_Ze)
            
            # new_e, ind, b, b_prime = updateK_MeanField(curr_e)
            # new_Ze =  updateZe_multi(curr_Ze, ind, [b for i in ind],[ b_prime for i in ind])  #更新扩展组标签Ze
            # new_matO, new_matN = updateMat_multi(new_Ze)
            
            # logNew = self.logpostpdf(newe, knowp)
            logNew = logpostpdf_fast(new_matO, new_matN)  #节点移动后的对数后验分布
            logdiff = logNew - logPost[i - 1]                  #计算新的后验分布与前一次后验分布的差值
            tmp = np.random.uniform(0, 1)                      #选取随机数
    
            # 判断是否接受这次移动，保存当前组标签和对数后验分不
            if tmp < np.exp(xi * logdiff):
                curr_e, curr_Ze = new_e, new_Ze
                curr_matO, curr_matN = new_matO, new_matN
                logPost.append(logNew)
    
            else:
                logPost.append(logPost[i - 1])
    
            # if i % vsteps == 0:
            #     R = Rdiff(currente, self.c)
            #     mist = np.sum(R - np.diag(np.diag(R)))
            #     print("iteration: {}, mistake: {}".format(i, mist))
            #     if self.show:
            #         print(R)
            #     n_mis.append(mist)
    
    
        logPost = logPost
        label = curr_e
        n_mis = n_mis
        finalR = Rdiff(curr_e, c)
        return logPost, label, n_mis, finalR
    
    
    ## 运行主程序
    repeat = 50
    niter = niter * n
    niter
    
    ## main for classic_MHsampler
    logpost_save = []
    initR_each_repeat_save = []
    alginitR_save = []
    finalR_save = []
    finale_save = []
    finalRI_save = []
    
    
    dirmake = "./K" + str(K) + "n" + str(n) + "p" + str(p) + "q" + str(q) + "/"
    if not os.path.exists(dirmake):
        os.makedirs(dirmake)
    
    
    for repeat_mark in range(repeat):
        e0 = np.zeros((n,)).astype(int)  #生成n*n的0矩阵
        ind = np.random.permutation(n)  #乱序后的n
        #halfn = int(n / 2)  # half wrong
        halfn = 1
        e0[ind[:halfn]] = c[ind[:halfn]]  #选取部分点，变为c中对应位置的值，此部分为正确的值
        e0[ind[halfn:]] = np.random.randint(K, size=(n-halfn))  #选取剩余点，随机变为可能的值（放进任意组），此部分为随机摆放，可能正确
        alginitR = Rdiff(e0, c)  #更新错误矩阵
        alginitR_save.append(alginitR)
        print("initial alginitR: ", alginitR)
    
        logPostList = []
        misList = []
        # labelList = []
        count = 1
        RList = []
        initR = alginitR
        initR_save = []
    
        while np.sum(initR - np.diag(np.diag(initR))) > 0 and count < 5:
            initR = Rdiff(e0, c)
            print("initial mistake is {}".format(
                np.sum(initR - np.diag(np.diag(initR)))))
            initR_save.append(np.sum(initR - np.diag(np.diag(initR))))
    
            res = classic_MHsampler(e0, niter, knowp, vsteps)
            e0 = res[1]
            logPostList.append(res[0])
            misList.append(res[2])
            RList.append(res[3])
            count = count + 1
            
        initR_each_repeat_save.append([initR_save])
        finalR = Rdiff(e0, c)
        mist = np.sum(finalR - np.diag(np.diag(finalR)))
        
        finale = e0
        RI = RIindex(finale, c)
        
        finale_save.append(finale)
        finalRI_save.append(RI)
        logpost_save.append(logPostList)
        finalR_save.append(finalR)

        
    with open(dirmake + "output_repeat" + str(repeat_mark) + ".pkl", "wb") as out:
        pickle.dump([logpost_save, initR_each_repeat_save, alginitR_save, finalR_save, finale_save, finalRI_save], out)
    
        '''
        with open(dirmake + "input_repeat" + str(repeat_mark) + ".pkl", "wb") as out:
            pickle.dump(
                [K, n, niter, Pstar, alginitR], out)
    
        # save_figs
        with open(dirmake + "output_repeat" + str(repeat_mark) + ".pkl", "wb") as out:
            pickle.dump(
                [logPostList, mist, RList], out)


    diffp_logpost.append(logpost_save)
    diffp_initR_repeat.append(initR_each_repeat_save)
    diffp_alginitR.append(alginitR_save)
    diffp_finalR.append(finalR_save)
    

    with open(dirmake + "output_repeat_package" + ".pkl", "wb") as out:
        pickle.dump([diffp_logpost, diffp_initR_repeat, diffp_alginitR, diffp_finalR], out)
    '''

    
endtime = datetime.datetime.now()
print(endtime - starttime)










