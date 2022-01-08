# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class MyFrame1
###########################################################################

class MyFrame1 ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 1000,628 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		fgSizer2 = wx.FlexGridSizer( 0, 2, 0, 0 )
		fgSizer2.SetFlexibleDirection( wx.BOTH )
		fgSizer2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.sec_str = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 900,-1 ), 0 )
		fgSizer2.Add( self.sec_str, 0, wx.ALL, 5 )

		self.sea_botton = wx.Button( self, wx.ID_ANY, u"Search", wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer2.Add( self.sea_botton, 0, wx.ALL, 5 )

		res_boxChoices = []
		self.res_box = wx.ListBox( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 900,500 ), res_boxChoices, wx.LB_NEEDED_SB|wx.LB_SINGLE|wx.LB_SORT )
		fgSizer2.Add( self.res_box, 0, wx.ALL, 5 )

		bSizer2 = wx.BoxSizer( wx.VERTICAL )

		self.Add_bot = wx.Button( self, wx.ID_ANY, u"Add", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.Add_bot, 0, wx.ALL, 5 )

		self.delete_bot = wx.Button( self, wx.ID_ANY, u"Delete", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer2.Add( self.delete_bot, 0, wx.ALL, 5 )


		fgSizer2.Add( bSizer2, 1, wx.EXPAND, 5 )


		self.SetSizer( fgSizer2 )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.sea_botton.Bind( wx.EVT_BUTTON, self.Search )
		self.res_box.Bind( wx.EVT_LISTBOX_DCLICK, self.open_image )
		self.Add_bot.Bind( wx.EVT_BUTTON, self.add )
		self.delete_bot.Bind( wx.EVT_BUTTON, self.delete )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def Search( self, event ):
		event.Skip()

	def open_image( self, event ):
		event.Skip()

	def add( self, event ):
		event.Skip()

	def delete( self, event ):
		event.Skip()


