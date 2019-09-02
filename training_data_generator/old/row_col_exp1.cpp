#include <stdint.h>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

#include <fstream>
#include <sstream>
#include <sys/stat.h>

#define UNIQUE_VALUES 10;

using namespace std;

static uint32_t randUInt32()
{
    return rand() % UNIQUE_VALUES;
}

static bool fileExists(const std::string &file)
{
    struct stat buffer;
    return (stat(file.c_str(), &buffer) == 0);
}

struct colgroup0_t
{
    bool operator<(const colgroup0_t &str) const
    {
        return true;
    }

    void print()
    {
    }
};

struct colgroup1_t
{
    uint32_t attr1;

    colgroup1_t(uint32_t a1) : attr1(a1)
    {
    }

    colgroup1_t()
    {
        attr1 = randUInt32();
    }

    bool operator<(const colgroup1_t &str) const
    {
        return (attr1 < str.attr1);
    }

    void print()
    {
        cout << attr1;
    }
};

struct colgroup2_t
{
    uint32_t attr1;
    uint32_t attr2;

    colgroup2_t(uint32_t a1, uint32_t a2) : attr1(a1), attr2(a2)
    {
    }

    colgroup2_t()
    {
        attr1 = randUInt32();
        attr2 = randUInt32();
    }

    bool operator<(const colgroup2_t &str) const
    {
        return (attr1 < str.attr1) ||
               (attr1 == str.attr1 && attr2 < str.attr2);
    }

    void print()
    {
        cout << attr1 << "," << attr2;
    }
};

struct colgroup3_t
{
    uint32_t attr1;
    uint32_t attr2;
    uint32_t attr3;

    colgroup3_t(uint32_t a1, uint32_t a2, uint32_t a3) : attr1(a1), attr2(a2), attr3(a3)
    {
    }

    colgroup3_t()
    {
        attr1 = randUInt32();
        attr2 = randUInt32();
        attr3 = randUInt32();
    }

    bool operator<(const colgroup3_t &str) const
    {
        return (attr1 < str.attr1) ||
               (attr1 == str.attr1 && attr2 < str.attr2) ||
               (attr1 == str.attr1 && attr2 == str.attr2 && attr3 < str.attr3);
    }

    void print()
    {
        cout << attr1 << "," << attr2 << "," << attr3;
    }
};

struct colgroup4_t
{
    uint32_t attr1;
    uint32_t attr2;
    uint32_t attr3;
    uint32_t attr4;

    colgroup4_t(uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4) : attr1(a1), attr2(a2), attr3(a3), attr4(a4)
    {
    }

    colgroup4_t()
    {
        attr1 = randUInt32();
        attr2 = randUInt32();
        attr3 = randUInt32();
        attr4 = randUInt32();
    }

    bool operator<(const colgroup4_t &str) const
    {
        return (attr1 < str.attr1) ||
               (attr1 == str.attr1 && attr2 < str.attr2) ||
               (attr1 == str.attr1 && attr2 == str.attr2 && attr3 < str.attr3) ||
               (attr1 == str.attr1 && attr2 == str.attr2 && attr3 == str.attr3 && attr4 < str.attr4);
    }

    void print()
    {
        cout << attr1 << "," << attr2 << "," << attr3 << "," << attr4;
    }
};

template <typename T1, typename T2, typename T3, typename T4>
class Table
{
private:
    size_t m_rows;
    T1 *m_col1;
    T2 *m_col2;
    T3 *m_col3;
    T4 *m_col4;

public:
    Table(size_t rows) : m_rows(rows)
    {
        m_col1 = new T1[m_rows];
        m_col2 = new T2[m_rows];
        m_col3 = new T3[m_rows];
        m_col4 = new T4[m_rows];
    }

    void setCol1(size_t i, T1 val)
    {
        m_col1[i] = val;
    }

    T1 col1(size_t i)
    {
        return m_col1[i];
    }

    void setCol2(size_t i, T2 val)
    {
        m_col2[i] = val;
    }

    T2 col2(size_t i)
    {
        return m_col2[i];
    }

    void setCol3(size_t i, T3 val)
    {
        m_col3[i] = val;
    }

    T3 col3(size_t i)
    {
        return m_col3[i];
    }

    void setCol4(size_t i, T4 val)
    {
        m_col4[i] = val;
    }

    T4 col4(size_t i)
    {
        return m_col4[i];
    }

    size_t rows()
    {
        return m_rows;
    }

    void sort()
    {
        std::sort(m_col1, m_col1 + m_rows);
    }

    void print(size_t rows)
    {
        rows = (rows < m_rows) ? rows : m_rows;

        for (size_t i = 0; i < rows; ++i)
        {
            m_col1[i].print();
            cout << ",";
            m_col2[i].print();
            cout << ",";
            m_col3[i].print();
            cout << ",";
            m_col4[i].print();
            cout << endl;
        }
    }
};

int main(int argc, char const *argv[])
{
    std::string filename = "row_col_exp1.csv";
    if (!fileExists(filename))
    {
        std::ofstream result_file(filename, std::ios_base::app | std::ios_base::out);
        result_file << "table_size,selectivity,runtime_us,layout\n";
        result_file.flush();
    }
    std::ofstream result_file(filename, std::ios_base::app | std::ios_base::out);

    //table data
    size_t rows = 100;

    //row table
    Table<colgroup4_t, colgroup0_t, colgroup0_t, colgroup0_t> t1(rows);
    t1.sort();
    t1.print(10);
    cout << "--------" << endl;

    //column table
    Table<colgroup1_t, colgroup1_t, colgroup1_t, colgroup1_t> t2(rows);
    for (size_t i = 0; i < t1.rows(); ++i)
    {
        colgroup4_t val = t1.col1(i);
        t2.setCol1(i, colgroup1_t(val.attr1));
        t2.setCol2(i, colgroup1_t(val.attr2));
        t2.setCol3(i, colgroup1_t(val.attr3));
        t2.setCol4(i, colgroup1_t(val.attr4));
    }
    t2.print(10);
    cout << "--------" << endl;

    //hybrid table
    Table<colgroup2_t, colgroup2_t, colgroup0_t, colgroup0_t> t3(rows);
    for (size_t i = 0; i < t1.rows(); ++i)
    {
        colgroup4_t val = t1.col1(i);
        t3.setCol1(i, colgroup2_t(val.attr1, val.attr2));
        t3.setCol2(i, colgroup2_t(val.attr3, val.attr4));
    }
    t3.print(10);
    cout << "--------" << endl;

    return 0;
}