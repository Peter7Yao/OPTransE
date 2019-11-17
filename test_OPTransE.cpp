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
#include "Eigen/Dense"
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

typedef struct ans{
    double sum0;
    double sum1;
}Ans;

string folder, suffix;
char buf[1000];
int relation_num, entity_num, nThreads = 8;
map<string, int> relation2id, entity2id;
map<int, string> id2entity, id2relation;

vector<int> rel_type;

map<pair<string,int>,double>  path_confidence;

map<int, int> entity2num;

map<int, map<int, int> > left_entity, right_entity;
map<int, double> left_avg, right_avg;

int one2one_num=0, n2one_num=0, one2n_num=0, n2n_num=0;

int myrandom(int i) {
	return std::rand() % i;
}
double cmp(pair<int, Ans> a, pair<int, Ans> b) {
    if(a.second.sum1 == b.second.sum1)
    {
        return a.second.sum0 < b.second.sum0;
    }
    else
    {
        return a.second.sum1 < b.second.sum1;
    }
}

class OPTransE {

public:
	map<pair<int, int>, map<int, int> > train_triples, dev_test_triples;
	map<int, double> headTailSelector;
    map<pair<int,int>,vector<pair<vector<int>,double> > >fb_path;

	void add(int h, int r, int t) {
		kb_h.push_back(h);
		kb_r.push_back(r);
		kb_t.push_back(t);
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
    
    void addTest(int x,int y,int z, vector<pair<vector<int>,double> > path_list, int flag)
    {
        if (z!=-1)
        {
            if(flag)
            {
                test_fb_h.push_back(x);
                test_fb_r.push_back(z);
                test_fb_t.push_back(y);
            }
            dev_test_triples[make_pair(x,z)][y]=1;
        }
        if (path_list.size()>0)
            fb_path[make_pair(x,y)] = path_list;
    }

	void run(int n_in) {
		n = n_in;

		for (int i = 0; i < relation_num; i++) {
			headTailSelector[i] = 1000 * right_avg[i]
					/ (right_avg[i] + left_avg[i]);
		}

		relation_vec.resize(relation_num*2);
		for (int i = 0; i < relation_vec.size(); i++)
			relation_vec[i].resize(n);
		entity_vec.resize(entity_num);
		for (int i = 0; i < entity_vec.size(); i++)
			entity_vec[i].resize(n);

        string name = "after_training/trained";
        
        FILE* f1 = fopen((folder + name + ".entity2vec").c_str(), "r");
        for (int i = 0; i < entity_num; i++) {
            for (int ii = 0; ii < n; ii++)
                fscanf(f1, "%lf", &entity_vec[i][ii]);
        }
        fclose(f1);
        
        FILE* f2 = fopen((folder + name + ".relation2vec").c_str(), "r");
        for (int i = 0; i < relation_num*2; i++) {
            for (int ii = 0; ii < n; ii++)
                fscanf(f2, "%lf", &relation_vec[i][ii]);
        }
        fclose(f2);
        
        FILE* f3 = fopen((folder + name + ".W1").c_str(), "r");
        W1.resize(relation_num*2);
        for (int i = 0; i < relation_num*2; i++) {
            W1[i].resize(n, n);
            for (int jj = 0; jj < n; jj++) {
                for (int ii = 0; ii < n; ii++) {
                    fscanf(f3, "%lf", &W1[i](jj,ii));
                }
            }
        }
        fclose(f3);
        
        FILE* f4 = fopen((folder + name + ".W2").c_str(), "r");
        W2.resize(relation_num*2);
        for (int i = 0; i < relation_num*2; i++) {
            W2[i].resize(n, n);
            for (int jj = 0; jj < n; jj++) {
                for (int ii = 0; ii < n; ii++) {
                    fscanf(f4, "%lf", &W2[i](jj,ii));
                }
            }
        }
        fclose(f4);
        
        test_OPTransE();
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

	void test_OPTransE() {
        cout << "\tEvaluating on test set: " << endl;
        vector<double> linkPredictionValues;
        runEntityPrediction_Test(linkPredictionValues);
        cout << "\tRaw scores (MR, MRR, H@1, H@3, H@5, H@10) on test set: "
        << linkPredictionValues[0] << " "
        << linkPredictionValues[1] << " "
        << linkPredictionValues[2] << " "
        << linkPredictionValues[3] << " "
        << linkPredictionValues[4] << " "
        << linkPredictionValues[5] << endl;
        
        cout
        << "\tFiltered scores (MR, MRR, H@1, H@3, H@5, H@10) on test set:"
        << " " << linkPredictionValues[6] << " "
        << linkPredictionValues[7] << " "
        << linkPredictionValues[8] << " "
        << linkPredictionValues[9] << " "
        << linkPredictionValues[10] << " "
        << linkPredictionValues[11] << endl;
        
        cout
        << "\tHead scores (1-to-1, 1-to-N, N-to-1, N-to-N) on test set:"
        << " " << linkPredictionValues[12] << " "
        << linkPredictionValues[13] << " "
        << linkPredictionValues[14] << " "
        << linkPredictionValues[15] << endl;
        
        cout
        << "\tTail scores (1-to-1, 1-to-N, N-to-1, N-to-N) on test set:"
        << " " << linkPredictionValues[16] << " "
        << linkPredictionValues[17] << " "
        << linkPredictionValues[18] << " "
        << linkPredictionValues[19] << endl;
	}

	Ans getScore_OPTransE(int &e1, int &rel, int &e2, int flag) {
        Ans a;
        
        double sum = 0;
		VectorXd d = entity_vec[e1] * W1[rel] + relation_vec[rel]
				- entity_vec[e2] * W2[rel];
		if (L1_flag)
			sum += d.lpNorm<1>();
		else
			sum += d.squaredNorm();
        
        int h = e1;
        int l = e2;
        
        vector<pair<vector<int>,double> > path_list = fb_path[make_pair(h,l)];
        
        a.sum0 = sum;
        
        
        if (path_list.size()>0)
        {
            for (int path_id = 0; path_id<path_list.size(); path_id++)
            {
                vector<int> rel_path = path_list[path_id].first;
                
                if(rel_path.size() > 2 || (rel_path.size() == 1 && rel_path[0] == rel))
                    continue;
                
                double pr_path = 0;
                double pr = path_list[path_id].second;
                string  s;
                ostringstream oss;//创建一个流
                for (int ii=0; ii<rel_path.size(); ii++)
                {
                    oss<<rel_path[ii]<<" ";
                }
                s=oss.str();
                if (path_confidence.count(make_pair(s,rel))>0)
                    pr_path = path_confidence[make_pair(s,rel)];
                
                if(!(pr_path > 0))
                    continue;
                
                if(calc_path(rel,rel_path,e1,e2) < sum)
                {
                    sum = calc_path(rel,rel_path,e1,e2);
                }
            }
        }
        
        a.sum1 = sum;
        
        return a;
	}
    
    double calc_path(int rel,vector<int> rel_path, int e1, int e2)
    {
        double sum=0;
        if(rel_path.size() == 1)
        {
            VectorXd d = relation_vec[rel] - relation_vec[rel_path[0]];
            sum = d.lpNorm<1>();
        }
        else
        {
            MatrixXd T = W1[rel_path[1]].inverse() * W2[rel_path[0]];
            VectorXd d = entity_vec[e1] * W1[rel_path[0]] +  relation_vec[rel_path[0]] + relation_vec[rel_path[1]] * T - entity_vec[e2] * W2[rel_path[1]] * T;
            
            sum = d.lpNorm<1>();
        }
        
        return sum;
    }

	void runEntityPrediction_Test(vector<double> &evalValues) {

		vector<double> values;
		for (int i = 0; i < 20; i++)
			values.push_back(0.0);

		int testSize = test_fb_h.size();
		int bSize = testSize / nThreads;
		map<int, vector<double> > results;

#pragma omp parallel for num_threads(nThreads)
		for (int i = 0; i < nThreads; i++) {
			int start = i * bSize;
			int end = (i + 1) * bSize;
			results[i] = evalEntityPrediction(start, end,
					train_triples, dev_test_triples, test_fb_h, test_fb_t,
					test_fb_r);
		}

		results[nThreads] = evalEntityPrediction(nThreads * bSize,
				testSize, train_triples, dev_test_triples, test_fb_h, test_fb_t,
				test_fb_r);

		for (map<int, vector<double> >::iterator it = results.begin();
				it != results.end(); it++) {
			vector<double> temp = it->second;
			for (int i = 0; i < 20; i++)
				values[i] += temp[i];
		}

		evalValues = values;
	}

	vector<double> evalEntityPrediction(int start, int end,
			map<pair<int, int>, map<int, int> > triples,
			map<pair<int, int>, map<int, int> > dev_test_triples,
			vector<int> test_fb_h, vector<int> test_fb_t,
			vector<int> test_fb_r) {

		vector<double> values;
		for (int i = 0; i < 20; i++)
			values.push_back(0.0);

		double headMR = 0, tailMR = 0, headH10 = 0, headH1 = 0, headH3 = 0, headH5 = 0,
				tailH10 = 0, headMRR = 0, tailMRR = 0, tailH1 = 0, tailH3 = 0, tailH5 = 0;

		double filter_headMR = 0, filter_tailMR = 0, filter_headH10 = 0,

		filter_headH1 = 0, filter_headH3 = 0, filter_headH5 = 0, filter_tailH10 = 0,
				filter_headMRR = 0, filter_tailMRR = 0, filter_tailH1 = 0,
                filter_tailH3 = 0, filter_tailH5 = 0;
        
        double l_one2one=0,r_one2one=0;
        double l_n2one=0,r_n2one=0;
        double l_one2n=0,r_one2n=0;
        double l_n2n=0,r_n2n=0;

		for (int validid = start; validid < end; validid++) {
			printf(" %d \n", validid);

			int head = test_fb_h[validid];
			int tail = test_fb_t[validid];
			int rel = test_fb_r[validid];
            
			vector<pair<int, Ans> > scores;
			for (int i = 0; i < entity_num; i++) {
                Ans sim;
				sim = getScore_OPTransE(i, rel, tail, 0);
				scores.push_back(make_pair(i, sim));
			}
			sort(scores.begin(), scores.end(), cmp);

			int filter = 0;
			for (int i = 0; i < scores.size(); i++) {

				if ((triples[make_pair(scores[i].first, rel)].count(tail) == 0)
						&& (dev_test_triples[make_pair(scores[i].first, rel)].count(
								tail) == 0))
					filter += 1;

				if (scores[i].first == head) {

					headMR += (i + 1);
					headMRR += 1.0 / (i + 1);
					if (i == 0)
						headH1 += 1;
                    if (i < 3)
                        headH3 += 1;
					if (i < 5)
						headH5 += 1;
					if (i < 10)
						headH10 += 1;

					filter_headMR += (filter + 1);
					filter_headMRR += 1.0 / (filter + 1);
					if (filter == 0)
						filter_headH1 += 1;
                    if (filter < 3)
                        filter_headH3 += 1;
					if (filter < 5)
						filter_headH5 += 1;
					if (filter < 10)
                    {
                        filter_headH10 += 1;

                        if (rel_type[rel]==0)
                            l_one2one+=1;
                        else if (rel_type[rel]==1)
                            l_n2one+=1;
                        else if (rel_type[rel]==2)
                            l_one2n+=1;
                        else
                            l_n2n+=1;
                    }

					break;
				}
			}
			scores.clear();

			for (int i = 0; i < entity_num; i++) {
				Ans sim;
				sim = getScore_OPTransE(head, rel, i, 0);
				scores.push_back(make_pair(i, sim));
			}
			sort(scores.begin(), scores.end(), cmp);

			filter = 0;
			for (int i = 0; i < scores.size(); i++) {

				if ((triples[make_pair(head, rel)].count(scores[i].first) == 0)
						&& (dev_test_triples[make_pair(head, rel)].count(
								scores[i].first) == 0))
					filter += 1;

				if (scores[i].first == tail) {
                    
					tailMR += (i + 1);
					tailMRR += 1.0 / (i + 1);

					if (i == 0)
						tailH1 += 1;
                    if (i < 3)
                        tailH3 += 1;
					if (i < 5)
						tailH5 += 1;
					if (i < 10)
						tailH10 += 1;

					filter_tailMR += (filter + 1);
					filter_tailMRR += 1.0 / (filter + 1);
					if (filter == 0)
						filter_tailH1 += 1;
                    if (filter < 3)
                        filter_tailH3 += 1;
					if (filter < 5)
						filter_tailH5 += 1;
                    if (filter < 10)
                    {
                        filter_tailH10 += 1;
                        
                        if (rel_type[rel]==0)
                            r_one2one+=1;
                        else if (rel_type[rel]==1)
                            r_n2one+=1;
                        else if (rel_type[rel]==2)
                            r_one2n+=1;
                        else
                            r_n2n+=1;
                    }

					break;
				}
			}
		}

		double test_size = 1.0 * test_fb_h.size();

		values[0] = (headMR + tailMR) / (2 * test_size);
		values[1] = (headMRR + tailMRR) / (2 * test_size);
		values[2] = (headH1 + tailH1) / (2 * test_size);
        values[3] = (headH3 + tailH3) / (2 * test_size);
		values[4] = (headH5 + tailH5) / (2 * test_size);
		values[5] = (headH10 + tailH10) / (2 * test_size);
		values[6] = (filter_headMR + filter_tailMR) / (2 * test_size);
		values[7] = (filter_headMRR + filter_tailMRR) / (2 * test_size);
		values[8] = (filter_headH1 + filter_tailH1) / (2 * test_size);
        values[9] = (filter_headH3 + filter_tailH3) / (2 * test_size);
		values[10] = (filter_headH5 + filter_tailH5) / (2 * test_size);
		values[11] = (filter_headH10 + filter_tailH10) / (2 * test_size);
        
        values[12] = l_one2one / one2one_num;
        values[13] = l_one2n / one2n_num;
        values[14] = l_n2one / n2one_num;
        values[15] = l_n2n / n2n_num;
        values[16] = r_one2one / one2one_num;
        values[17] = r_one2n / one2n_num;
        values[18] = r_n2one / n2one_num;
        values[19] = r_n2n / n2n_num;

		return values;
	}
};

OPTransE optranse;
void readData() {
    FILE* f7 = fopen((folder + "n2n.txt").c_str(), "r");
    {
        double x,y;
        while (fscanf(f7,"%lf%lf",&x,&y)==2)
        {
            if (x<1.5)
            {
                if (y<1.5)
                    rel_type.push_back(0);
                else
                    rel_type.push_back(1);
                
            }
            else
                if (y<1.5)
                    rel_type.push_back(2);
                else
                    rel_type.push_back(3);
        }
    }
    fclose(f7);
    
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
    
    int kk = 0;
    FILE* f_kb = fopen((folder + "test_pra.txt").c_str(), "r");
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
            if(kk == 0)
                b.push_back(make_pair(rel_path,pr));
        }
        b.clear();
        
        if(kk == 0)
        {
            optranse.addTest(e1, e2, rel, b, 1);
            
            if (rel_type[rel]==0)
                one2one_num+=1;
            else if (rel_type[rel]==1)
                n2one_num+=1;
            else if (rel_type[rel]==2)
                one2n_num+=1;
            else
                n2n_num+=1;
        }
        
        kk = 1 - kk;
    }
    
    fclose(f_kb);
    
	f_kb = fopen((folder + "train.txt").c_str(), "r");
	while (fscanf(f_kb, "%s", buf) == 1) {
		string head = buf; //left entity

		fscanf(f_kb, "%s", buf);
		string rel = buf;	//relation

		fscanf(f_kb, "%s", buf);
		string tail = buf;	//right entity

		left_entity[relation2id[rel]][entity2id[head]]++;
		right_entity[relation2id[rel]][entity2id[tail]]++;

		//Input: left/head entity, right/tail entity, relation
		optranse.add(entity2id[head], relation2id[rel], entity2id[tail]);
	}
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
    
    FILE* f_path = fopen((folder + "path2.txt").c_str(), "r");
    while (fscanf(f_path,"%s",buf)==1)
    {
        string s1=buf;
        fscanf(f_path,"%s",buf);
        string s2=buf;

        int e1 = entity2id[s1];
        int e2 = entity2id[s2];
        fscanf(f_path,"%d",&x);
        vector<pair<vector<int>,double> > b;
        b.clear();
        for (int i = 0; i<x; i++)
        {
            int y,z;
            vector<int> rel_path;
            rel_path.clear();
            fscanf(f_path,"%d",&y);
            for (int j=0; j<y; j++)
            {
                fscanf(f_path,"%d",&z);
                rel_path.push_back(z);
            }
            double pr;
            fscanf(f_path,"%lf",&pr);
            b.push_back(make_pair(rel_path,pr));
        }
        optranse.addTest(e1,e2,-1,b,1);
    }
    fclose(f_path);
    
    FILE* f_confidence = fopen((folder + "confidence.txt").c_str(), "r");
    while (fscanf(f_confidence,"%d",&x)==1)
    {
        string s = "";
        for (int i=0; i<x; i++)
        {
            fscanf(f_confidence,"%s",buf);
            s = s + string(buf)+" ";
        }
        fscanf(f_confidence,"%d",&x);
        for (int i=0; i<x; i++)
        {
            int y;
            double pr;
            fscanf(f_confidence,"%d%lf",&y,&pr);
            path_confidence[make_pair(s,y)] = pr;
        }
    }
    fclose(f_confidence);
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
    
    nThreads = 8;
    if ((i = ArgPos((char *) "-nthreads", argc, argv)) > 0) {
        nThreads = atoi(argv[i + 1]);
    }

    readData();
    
    optranse.run(n);
}

