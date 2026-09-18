# -*- coding: utf-8 -*-
"""把 flowcharts 目录下的 pptx 全部经 WPS 演示导出为 PDF。"""
import os, sys, time
import pythoncom
import win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pythoncom.CoInitialize()
    app = win32com.client.Dispatch("KWPP.Application")
    try:
        app.Visible = False
    except Exception:
        pass
    names = sys.argv[1:] or sorted(f for f in os.listdir(HERE) if f.endswith(".pptx"))
    for name in names:
        src = os.path.join(HERE, name)
        dst = os.path.join(os.path.dirname(HERE), "figures", name.replace(".pptx", ".pdf"))
        pres = app.Presentations.Open(src, 0, 0, 0)
        pres.SaveAs(dst, 32)
        pres.Close()
        print("->", dst, os.path.exists(dst))
    app.Quit()


if __name__ == "__main__":
    main()
