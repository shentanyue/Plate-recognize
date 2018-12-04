// Python_C.cpp : 定义控制台应用程序的入口点。
//
#include <iostream>
#include <Python.h>
#include <ctime>
#include <time.h>
using namespace std;
void recognition()
{
	const char* picpath = "/home/shenty/eclipse-workspace/test/src/Generating_data/plate/浙A9LL6P.jpg";
	int rect[4] = { 0,0,272,72 };
	Py_Initialize();//调用Py_Initialize()进行初始化
	if (!Py_IsInitialized())
		return;
	PyRun_SimpleString("import os,sys");
	PyRun_SimpleString("sys.path.append('./')");
	PyObject * pModule = NULL;
	PyObject * pFunc = NULL;
	PyObject * pRetVal = NULL;
	PyObject * pClass = NULL;
	PyObject * pArg = NULL;
	PyObject * pObject = NULL;
	PyObject * pParm = NULL;
	pModule = PyImport_ImportModule("demo");//调用的Python文件名
	if (!pModule)
	{
		cout << "打开python文件失败";
		return;
	}
	pClass = PyObject_GetAttrString(pModule, "recognition");
	pArg = PyTuple_New(1);
	PyTuple_SetItem(pArg, 0, Py_BuildValue("s","model/ocr_plate_all_gru.h5"));
	pObject = PyEval_CallObject(pClass, pArg);

	struct timeval tv;
	double a,b,c=0;

	pFunc = PyObject_GetAttrString(pObject, "run_demo");//调用的函数名
	if (!pFunc)
	{
		cout << "无此方法";
		return;
	}
	PyObject *pyParams = PyList_New(0);//创建列表
	PyList_Append(pyParams, Py_BuildValue("i", rect[0]));
	PyList_Append(pyParams, Py_BuildValue("i", rect[1]));
	PyList_Append(pyParams, Py_BuildValue("i", rect[2]));
	PyList_Append(pyParams, Py_BuildValue("i", rect[3]));

	pParm = PyTuple_New(2);//函数调用的参数传递均是以元组的形式打包的,2表示参数个数
	PyTuple_SetItem(pParm, 0, Py_BuildValue("s", picpath));//0--序号,s-zifuchuan
	//PyTuple_SetItem(pParm, 1, Py_BuildValue("[i,i,i,i]",0,0,220,72));//1--序号
	PyTuple_SetItem(pParm, 1, pyParams);//1--序号
	//PyObject *pReturn = NULL;
	//pReturn =
	for(int i=0;i<10;i++)
	{
		gettimeofday(&tv,NULL);
		a=tv.tv_sec*1000+tv.tv_usec/1000;
		PyEval_CallObject(pFunc, pParm);//调用函数
		gettimeofday(&tv,NULL);
		b=tv.tv_sec*1000+tv.tv_usec/1000;
		c=b-a;
		printf("%lf ms\n",(b-a));
	}

	Py_Finalize();//调用Py_Finalize,和Py_Initialize相对应的.
}

int main(int argc, char** argv)
{
	cout << "调用demo.py中的recognition函数..." << endl;
	recognition();
	getchar();
	return 0;
}
