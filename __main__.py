import wx
import GUI
import image_process
import os


def openFileDialog(style=wx.FD_MULTIPLE, message="选择文件", defaultDir=os.getcwd(), wildcard="All files(*.*)|*.*"):
    frame = wx.Frame(None, title="", pos=(0, 0), size=(100, 100))
    dlg = wx.FileDialog(parent=frame, message=message,
                        defaultDir=defaultDir,
                        style=style,
                        wildcard=wildcard)
    if dlg.ShowModal() == wx.ID_OK:
        if style == wx.FD_MULTIPLE:
            return dlg.GetPaths()
        return dlg.GetPath()
    dlg.Destroy()
    return None


class mainWin(GUI.MyFrame1):
    def Search(self, event):
        self.res_box.Clear()
        str = self.sec_str.GetValue()
        dict = {}
        if len(str) == 0:
            dict = image_process.get_list([])
        else:
            dict = image_process.get_list(str.split(' '))
        for k, v in dict.items():
            self.res_box.Append(k+': '+v)

    def open_image(self, event):
        index = self.res_box.GetSelection()
        str = self.res_box.GetString(index)
        words = str.split(':')
        image_process.open_image(words[0]+':'+words[1])

    def add(self, event):
        image_process.add_images(openFileDialog(wildcard="JPEG|*.jpg|BMP|*.bmp|PNG|*.png"))
        self.Search(event)

    def delete(self, event):
        index = self.res_box.GetSelection()
        if index == wx.NOT_FOUND:
            return
        str = self.res_box.GetString(index)
        words = str.split(':')
        image_process.delete_images([words[0] + ':' + words[1]])
        self.Search(event)

app = wx.App()
main_win = mainWin(None)
main_win.Show()
app.MainLoop()