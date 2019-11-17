#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include <algorithm>
#include<cstdlib>
#include<sstream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

#define pi 3.1415926535897932384626433832795

double rand(double min, double max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}
double normal(double x, double miu, double sigma) {
	return 1.0 / sqrt(2 * pi) / sigma
			* exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma));
}
double sigmod(double x) {
	return 1.0 / (1 + exp(-2 * x));
}
double randn(double miu, double sigma, double min, double max) {
	double x, y, dScope;
	do {
		x = rand(min, max);
		y = normal(x, miu, sigma);
		dScope = rand(0.0, normal(miu, miu, sigma));
	} while (dScope > y);
	return x;
}

double sqr(double x) {
	return x * x;
}

double vec_len(vector<double> &a) {
	double res = 0;
	for (int i = 0; i < a.size(); i++)
		res += a[i] * a[i];
	res = sqrt(res);
	return res;
}

string folder, suffix;
int nepoch = 2000;
char buf[2000];
int relation_num, entity_num, nThreads = 8;
map<string, int> relation2id, entity2id;
map<int, string> id2entity, id2relation;

map<vector<int>,string> path2s;
map<pair<string,int>,double>  path_confidence;
vector<vector<pair<int,int> > > path;

map<int, int> entity2num;

map<int, map<int, int> > left_entity, right_entity;
map<int, double> left_avg, right_avg;

int myrandom(int i) {
	return std::rand() % i;
}
double cmp(pair<int, double> a, pair<int, double> b) {
	return a.second < b.second;
}

class OPTransE {

public:
	map<pair<int, int>, map<int, int> > train_triples, dev_test_triples;
	map<int, double> headTailSelector;

	void add(int h, int r, int t) {
		kb_h.push_back(h);
		kb_r.push_back(r);
		kb_t.push_back(t);
		train_triples[make_pair(h, r)][t] = 1;
	}
    
    void add(int h, int r, int t, vector<pair<vector<int>,double> > path_list)
    {
        kb_h.push_back(h);
        kb_r.push_back(r);
        kb_t.push_back(t);
        fb_path.push_back(path_list);
        train_triples[make_pair(h, r)][t] = 1;
    }

	void addDev(int h, int r, int t) {
		dev_fb_h.push_back(h);
		dev_fb_r.push_back(r);
		dev_fb_t.push_back(t);
		dev_test_triples[make_pair(h, r)][t] = 1;
	}

	void addTest(int h, int r, int t) {
		test_fb_h.push_back(h);
		test_fb_r.push_back(r);
		test_fb_t.push_back(t);
		dev_test_triples[make_pair(h, r)][t] = 1;
	}

	void run(int n_in, double rate_in, double margin_in, bool l1_in) {
		n = n_in;
		rate = rate_in;
		margin = margin_in;
		L1_flag = l1_in;

		for (int i = 0; i < relation_num; i++) {
			headTailSelector[i] = 1000 * right_avg[i]
					/ (right_avg[i] + left_avg[i]);
		}

		relation_vec.resize(relation_num);
		for (int i = 0; i < relation_vec.size(); i++)
			relation_vec[i].resize(n);
		entity_vec.resize(entity_num);
		for (int i = 0; i < entity_vec.size(); i++)
			entity_vec[i].resize(n);

        string name = "init/init";
        
        FILE* f1 = fopen((folder + name + ".entity2vec").c_str(), "r");
        for (int i = 0; i < entity_num; i++) {
            for (int ii = 0; ii < n; ii++)
                fscanf(f1, "%lf", &entity_vec[i][ii]);
            norm(entity_vec[i]);
        }
        fclose(f1);
        
        FILE* f2 = fopen((folder + name + ".relation2vec").c_str(), "r");
        for (int i = 0; i < relation_num; i++) {
            for (int ii = 0; ii < n; ii++)
                fscanf(f2, "%lf", &relation_vec[i][ii]);
        }
        fclose(f2);
        
        FILE* f3 = fopen((folder + name + ".W1").c_str(), "r");
        W1.resize(relation_num);
        for (int i = 0; i < relation_num; i++) {
            W1[i].resize(n, n);
            for (int jj = 0; jj < n; jj++) {
                for (int ii = 0; ii < n; ii++) {
                    fscanf(f3, "%lf", &W1[i](jj,ii));
                }
            }
        }
        fclose(f3);
        
        FILE* f4 = fopen((folder + name + ".W2").c_str(), "r");
        W2.resize(relation_num);
        for (int i = 0; i < relation_num; i++) {
            W2[i].resize(n, n);
            for (int jj = 0; jj < n; jj++) {
                for (int ii = 0; ii < n; ii++) {
                    fscanf(f4, "%lf", &W2[i](jj,ii));
                }
            }
        }
        fclose(f4);
        
        optimize_OPTransE();

	}

private:
	int n;
	double cost, rate, margin;
	vector<int> kb_h, kb_t, kb_r;
	vector<int> dev_fb_h, dev_fb_t, dev_fb_r;
	vector<int> test_fb_h, test_fb_t, test_fb_r;
	vector<RowVectorXd> relation_vec, entity_vec, relation_tmp, entity_tmp;
	vector<MatrixXd> W1, W1_tmp, W2, W2_tmp;
	bool L1_flag = true;
    int isInit = 1;
    vector<vector<pair<vector<int>,double> > >fb_path;

	void norm(RowVectorXd &a) {
		if (a.norm() > 1)
			a.normalize();
	}

	void norm(RowVectorXd &a, MatrixXd &A) {
		while (true) {
			double x = (a * A).norm();
			if (x > 1) {
				for (int ii = 0; ii < n; ii++) {
					double tmp = A.col(ii).dot(a);
					for (int jj = 0; jj < n; jj++) {
						A(jj, ii) -= rate * tmp * a[jj];
						a[jj] -= rate * tmp * A(jj, ii);
					}
				}
			} else
				break;
		}
	}

	void optimize_OPTransE() {

		cout.precision(10);

		FILE* fLog = fopen((folder + "OPTransE" + suffix + ".log.txt").c_str(),
				"w");

		cout
				<< "Optimize entity vectors, relation vectors and relation matrices:"
				<< endl;
		fprintf(fLog, "%s\n",
				"Optimize entity vectors, relation vectors and relation matrices:");

		for (int epoch = 0; epoch < nepoch; epoch++) {

			cost = 0;

			relation_tmp = relation_vec;
			entity_tmp = entity_vec;
			W1_tmp = W1;
			W2_tmp = W2;

			for (int i = 0; i < kb_h.size(); ++i) {
				int sampledEn = rand() % entity_num;

				int head = kb_h[i], tail = kb_t[i], rel = kb_r[i];

				double pr = headTailSelector[rel];
                
				if (rand() % 1000 < pr) {

					while (train_triples[make_pair(head, rel)].count(sampledEn)
							> 0)
						sampledEn = rand() % entity_num;

					updateParas_OPTransE(head, rel, tail, head, rel, sampledEn);

					norm(relation_tmp[rel]);
					norm(entity_tmp[head]);
					norm(entity_tmp[tail]);
					norm(entity_tmp[sampledEn]);
					norm(entity_tmp[head], W1_tmp[rel]);
					norm(entity_tmp[tail], W2_tmp[rel]);
					norm(entity_tmp[sampledEn], W2_tmp[rel]);

				} else {
					while (train_triples[make_pair(sampledEn, rel)].count(tail)
							> 0)
						sampledEn = rand() % entity_num;

					updateParas_OPTransE(head, rel, tail, sampledEn, rel, tail);

					norm(relation_tmp[rel]);
					norm(entity_tmp[head]);
					norm(entity_tmp[tail]);
					norm(entity_tmp[sampledEn]);
					norm(entity_tmp[head], W1_tmp[rel]);
					norm(entity_tmp[tail], W2_tmp[rel]);
					norm(entity_tmp[sampledEn], W1_tmp[rel]);
				}
                
                
                // train path
                
                if (fb_path[i].size()>0)
                {
                    for (int path_id = 0; path_id<fb_path[i].size(); path_id++)
                    {
                        vector<int> rel_path = fb_path[i][path_id].first;
                        string  s = "";
                        if (path2s.count(rel_path)==0)
                        {
                            ostringstream oss;
                            for (int ii=0; ii<rel_path.size(); ii++)
                            {
                                oss<<rel_path[ii]<<" ";
                            }
                            s=oss.str();//
                            path2s[rel_path] = s;
                        }
                        s = path2s[rel_path];
                        
                        double pr = fb_path[i][path_id].second;
                        double pr_path = 0;
                        if (path_confidence.count(make_pair(s,rel))>0)
                            pr_path = path_confidence[make_pair(s,rel)];
                        
                        if(!(pr_path > 0))
                            continue;

                        if(rel_path.size() == 1)
                        {
                            int sampledEn = rand() % entity_num;
                            if (rand() % 1000 < pr)
                            {
                                while (train_triples[make_pair(head, rel)].count(sampledEn) > 0)
                                    sampledEn = rand() % entity_num;
                                train_path(head, tail, head, sampledEn, rel_path, 4.5, pr*pr_path);
                            }
                            else
                            {
                                while (train_triples[make_pair(sampledEn, rel)].count(tail) > 0)
                                    sampledEn = rand() % entity_num;
                                train_path(head, tail, sampledEn, tail, rel_path, 4.5, pr*pr_path);
                            }
                            norm(relation_tmp[rel_path[0]]);
                        }
                        else if(rel_path.size() == 2)
                        {
                            int sampledEn = rand() % entity_num;
                            if (rand() % 1000 < pr)
                            {
                                while (train_triples[make_pair(head, rel)].count(sampledEn) > 0)
                                    sampledEn = rand() % entity_num;
                                train_path_long(head, tail, head, sampledEn, rel_path, 5.0, pr*pr_path);
                            }
                            else
                            {
                                while (train_triples[make_pair(sampledEn, rel)].count(tail) > 0)
                                    sampledEn = rand() % entity_num;
                                train_path_long(head, tail, sampledEn, tail, rel_path, 5.0, pr*pr_path);
                            }
                            norm(relation_tmp[rel_path[0]]);
                            norm(relation_tmp[rel_path[1]]);
                        }
                        
                    }
                }

				relation_vec[rel] = relation_tmp[rel];
				W1[rel] = W1_tmp[rel];
				W2[rel] = W2_tmp[rel];
				entity_vec[head] = entity_tmp[head];
				entity_vec[tail] = entity_tmp[tail];
				entity_vec[sampledEn] = entity_tmp[sampledEn];

			}

			relation_vec = relation_tmp;
			entity_vec = entity_tmp;
			W1 = W1_tmp;
			W2 = W2_tmp;

			cout << "\tepoch " << epoch << " : " << cost << endl;
			fprintf(fLog, "\t%s %d : %.6lf\n", "---\nepoch ", epoch, cost);
            
			if ((epoch + 1) % nepoch == 0) {
				write(epoch);
			}
		}
		fclose(fLog);
	}

	void updateParas_OPTransE(int &e1_a, int &rel_a, int &e2_a, int &e1_b,
			int &rel_b, int &e2_b) {
		VectorXd temp1, temp2;
		double sum1 = getScore_OPTransE(e1_a, rel_a, e2_a, temp1);
		double sum2 = getScore_OPTransE(e1_b, rel_b, e2_b, temp2);
		if (sum1 + margin > sum2) {
			cost += margin + sum1 - sum2;
			SGDupdate_OPTransE(e1_a, rel_a, e2_a, temp1, 1);
			SGDupdate_OPTransE(e1_b, rel_b, e2_b, temp2, -1);
		}
	}
    
    void train_path(int e1_a, int e2_a, int e1_b, int e2_b, vector<int> rel_path, double margin, double x)
    {
        double sum1 = calc_path(e1_a, rel_path, e2_a);
        double sum2 = calc_path(e1_b, rel_path, e2_b);
        double lambda = 0.01;
        if (sum1+margin>sum2)
        {
            cost += x*lambda*(margin + sum1 - sum2);
            gradient_path(e1_a, rel_path, e2_a, x*lambda);
            gradient_path(e1_b, rel_path, e2_b, -x*lambda);
        }
    }
    
    void train_path_long(int e1_a, int e2_a, int e1_b, int e2_b, vector<int> rel_path, double margin, double x)
    {
        double sum1 = calc_path_long(e1_a, rel_path, e2_a);
        double sum2 = calc_path_long(e1_b, rel_path, e2_b);
        double lambda = 0.01;
        if (sum1 + margin > sum2)
        {
            cost += x*lambda*(margin + sum1 - sum2);
            gradient_path_long(e1_a, rel_path, e2_a, x*lambda);
            gradient_path_long(e1_b, rel_path, e2_b, -x*lambda);
        }
    }

	double getScore_OPTransE(int &e1, int &rel, int &e2, VectorXd &d) {
		d = entity_vec[e1] * W1[rel] + relation_vec[rel]
				- entity_vec[e2] * W2[rel];
		if (L1_flag)
			return d.lpNorm<1>();
		else
			return d.squaredNorm();
	}

	double getScore_OPTransE(int &e1, int &rel, int &e2) {
		VectorXd d = entity_vec[e1] * W1[rel] + relation_vec[rel]
				- entity_vec[e2] * W2[rel];
		if (L1_flag)
			return d.lpNorm<1>();
		else
			return d.squaredNorm();
	}
    
    double calc_path(int e1, vector<int> rel_path, int e2)
    {
        double sum=0;
        VectorXd d = d = entity_vec[e1] * W1[rel_path[0]] + relation_vec[rel_path[0]] - entity_vec[e2] * W2[rel_path[0]];
        sum = d.lpNorm<1>();
        return sum;
    }
    
    double calc_path_long(int e1, vector<int> rel_path, int e2)
    {
        double sum=0;
        MatrixXd T = W1[rel_path[1]].inverse() * W2[rel_path[0]];
        VectorXd d = entity_vec[e1] * W1[rel_path[0]] +  relation_vec[rel_path[0]] + relation_vec[rel_path[1]] * T - entity_vec[e2] * W2[rel_path[1]] * T;
        sum = d.lpNorm<1>();
        return sum;
    }

	void SGDupdate_OPTransE(int &e1, int &rel, int &e2, VectorXd &d,
			int isCorrect) {
		for (int i = 0; i < n; i++) {
			double x = 2 * d[i];
			if (L1_flag)
				if (x > 0)
					x = 1;
				else
					x = -1;

			double tmp = isCorrect * rate * x;

			W1_tmp[rel].col(i) -= tmp * entity_vec[e1].transpose();
			W2_tmp[rel].col(i) += tmp * entity_vec[e2].transpose();
			entity_tmp[e1] -= tmp * W1[rel].col(i).transpose();
			entity_tmp[e2] += tmp * W2[rel].col(i).transpose();
			relation_tmp[rel][i] -= tmp;
		}
	}
    
    void gradient_path(int e1, vector<int> rel_path, int e2, double delta)
    {
        VectorXd d = d = entity_vec[e1] * W1[rel_path[0]] + relation_vec[rel_path[0]] - entity_vec[e2] * W2[rel_path[0]];
        
        for (int i = 0; i < n; i++)
        {
            double x = 2 * d[i];
            if (L1_flag)
                if (x > 0)
                    x = 1;
                else
                    x = -1;
            
            double tmp = delta * rate * x;

            relation_tmp[rel_path[0]][i] -= tmp;
        }
    }
    
    void gradient_path_long(int e1, vector<int> rel_path, int e2, double delta)
    {
        MatrixXd T = W1[rel_path[1]].inverse() * W2[rel_path[0]];
        VectorXd d = entity_vec[e1] * W1[rel_path[0]] +  relation_vec[rel_path[0]] + relation_vec[rel_path[1]] * T - entity_vec[e2] * W2[rel_path[1]] * T;
        MatrixXd M_1 = W2[rel_path[1]] * T;
        
        for (int i = 0; i < n; i++)
        {
            double x = d[i];
            if (L1_flag)
            {
                if (x > 0)
                    x = 1;
                else
                    x = -1;
            }
            
            double tmp = delta * rate * x;
            
            relation_tmp[rel_path[0]][i] -= tmp;
            relation_tmp[rel_path[1]] -= tmp * T.col(i).transpose();
        }
    }

    
	void write(int epoch) {
		ostringstream ss;
		ss << (epoch + 1);
		
        FILE* f1 = fopen(
                         (folder + "OPTransE" + suffix + ".e" + ss.str()
                          + ".relation2vec").c_str(), "w");
        FILE* f2 = fopen(
                         (folder + "OPTransE" + suffix + ".e" + ss.str()
                          + ".entity2vec").c_str(), "w");
        
        for (int i = 0; i < relation_num; i++) {
            for (int ii = 0; ii < n; ii++)
                fprintf(f1, "%.6lf\t", relation_vec[i][ii]);
            fprintf(f1, "\n");
        }
        for (int i = 0; i < entity_num; i++) {
            for (int ii = 0; ii < n; ii++)
                fprintf(f2, "%.6lf\t", entity_vec[i][ii]);
            fprintf(f2, "\n");
        }
        
        fclose(f1);
        fclose(f2);
        
        FILE* f3 =
        fopen(
              (folder + "OPTransE" + suffix + ".e" + ss.str()
               + ".W1").c_str(), "w");
        FILE* f4 =
        fopen(
              (folder + "OPTransE" + suffix + ".e" + ss.str()
               + ".W2").c_str(), "w");
        for (int i = 0; i < relation_num; i++)
            for (int jj = 0; jj < n; jj++) {
                for (int ii = 0; ii < n; ii++) {
                    fprintf(f3, "%.6lf\t", W1[i](jj, ii));
                }
                fprintf(f3, "\n");
            }
        
        for (int i = 0; i < relation_num; i++)
            for (int jj = 0; jj < n; jj++) {
                for (int ii = 0; ii < n; ii++) {
                    fprintf(f4, "%.6lf\t", W2[i](jj, ii));
                }
                fprintf(f4, "\n");
            }
        
        fclose(f3);
        fclose(f4);
	}

};

OPTransE optranse;

void readData() {
	FILE* f1 = fopen((folder + "entity2id.txt").c_str(), "r");
	FILE* f2 = fopen((folder + "relation2id.txt").c_str(), "r");
	int x;
	while (fscanf(f1, "%s%d", buf, &x) == 2) {
		string st = buf;
		entity2id[st] = x;
		id2entity[x] = st;
		entity_num++;
	}
	while (fscanf(f2, "%s%d", buf, &x) == 2) {
		string st = buf;
		relation2id[st] = x;
		id2relation[x] = st;
		relation_num++;
	}
    
    FILE* f_kb = fopen((folder + "train_pra.txt").c_str(), "r");
    while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_kb,"%s",buf);
        string s2=buf;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        int rel;
        fscanf(f_kb,"%d",&rel);
        fscanf(f_kb,"%d",&x);
        vector<pair<vector<int>,double> > b;
        b.clear();
        for (int i = 0; i<x; i++)
        {
            int y,z;
            vector<int> rel_path;
            rel_path.clear();
            fscanf(f_kb,"%d",&y);
            for (int j=0; j<y; j++)
            {
                fscanf(f_kb,"%d",&z);
                rel_path.push_back(z);
            }
            double pr;
            fscanf(f_kb,"%lf",&pr);
            b.push_back(make_pair(rel_path,pr));
        }
        
        left_entity[rel][e1]++;
        right_entity[rel][e2]++;
        
        optranse.add(e1,rel,e2,b);
    }
    
    relation_num *= 2;
    
	for (int i = 0; i < relation_num; i++) {
		double sum1 = 0, sum2 = 0;
		for (map<int, int>::iterator it = left_entity[i].begin();
				it != left_entity[i].end(); it++) {
			sum1++;
			sum2 += it->second;
		}
		left_avg[i] = sum2 / sum1;
	}
	for (int i = 0; i < relation_num; i++) {
		double sum1 = 0, sum2 = 0;
		for (map<int, int>::iterator it = right_entity[i].begin();
				it != right_entity[i].end(); it++) {
			sum1++;
			sum2 += it->second;
		}
		right_avg[i] = sum2 / sum1;
	}
	cout << "#relations = " << relation_num << endl;
	cout << "#entities = " << entity_num << endl;
	fclose(f_kb);

	f_kb = fopen((folder + "valid.txt").c_str(), "r");
	while (fscanf(f_kb, "%s", buf) == 1) {
		string head = buf;

		fscanf(f_kb, "%s", buf);
		string rel = buf;

		fscanf(f_kb, "%s", buf);
		string tail = buf;

		optranse.addDev(entity2id[head], relation2id[rel], entity2id[tail]);
	}
	fclose(f_kb);

	f_kb = fopen((folder + "test.txt").c_str(), "r");
	while (fscanf(f_kb, "%s", buf) == 1) {
		string head = buf;

		fscanf(f_kb, "%s", buf);
		string rel = buf;

		fscanf(f_kb, "%s", buf);
		string tail = buf;

		optranse.addTest(entity2id[head], relation2id[rel], entity2id[tail]);
	}
	fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}

int main(int argc, char**argv) {
	srand((unsigned) time(NULL));

	int i = 0;

	if ((i = ArgPos((char *) "-datapath", argc, argv)) > 0) {
		folder = argv[i + 1];
	}

	int n = 100;
	if ((i = ArgPos((char *) "-size", argc, argv)) > 0) {
		n = atoi(argv[i + 1]);
		suffix += ".s" + string(argv[i + 1]);
	}

	double rate = 0.0005;
	if ((i = ArgPos((char *) "-lrate", argc, argv)) > 0) {
		rate = atof(argv[i + 1]);
		suffix += ".r" + string(argv[i + 1]);
	}

	double margin = 4.0;
	if ((i = ArgPos((char *) "-margin", argc, argv)) > 0) {
		margin = atof(argv[i + 1]);
		suffix += ".m" + string(argv[i + 1]);
	}

	bool l1 = 1;
	if ((i = ArgPos((char *) "-l1", argc, argv)) > 0) {
		l1 = atoi(argv[i + 1]);
		suffix += ".l1_" + string(argv[i + 1]);
	}

	nepoch = 2000;
	if ((i = ArgPos((char *) "-nepoch", argc, argv)) > 0) {
		nepoch = atoi(argv[i + 1]);
	}

	nThreads = 8;
	if ((i = ArgPos((char *) "-nthreads", argc, argv)) > 0) {
		nThreads = atoi(argv[i + 1]);
	}

	cout << "Dataset: " << folder << endl;
	cout << "Number of epoches: " << nepoch << endl;
	cout << "Vector size: " << n << endl;
	cout << "Margin: " << margin << endl;
	cout << "L1-norm: " << l1 << endl;
	cout << "SGD learing rate: " << rate << endl;
	cout << "nThreads: " << nThreads << endl;

	readData();

	optranse.run(n, rate, margin, l1);
}

