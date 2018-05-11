// class for reading the sparse matrix data
// for both user confidence matrix and item confidence matrix
// user matrix:
// number_of_items confidence_item1 confidence_item2 ...
// item matrix:
// number_of_users confidence_user1 confidence_user2 ...

#ifndef DATA_H
#define DATA_H

#include <vector>

using namespace std;

class c_double_data {
public:
  c_double_data();
  ~c_double_data();
  void read_data(const char * data_filename, int OFFSET=0);
public:
  vector<double*> m_vec_data;
  vector<double> m_vec_len;
};

class c_data {
public:
  c_data();
  ~c_data();
  void read_data(const char * data_filename, int OFFSET=0);
public:
  vector<int*> m_vec_data;
  vector<int> m_vec_len;
};

#endif // DATA_H
