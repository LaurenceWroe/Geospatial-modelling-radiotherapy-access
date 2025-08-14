import tkinter
import customtkinter




def start_download():
    try:
        yt_link = link.get()
        yt_object = YouTube(yt_link)
        views = yt_object.views
        print(views)
    except:
        print("bad")

# def startDownload():
#     try:
#         ytLink = url_var.get()
#         ytObject = YouTube(ytLink)
#         video = ytObject.views()
#         print(video)
#         # video.download()
#         # print("Download Complete")
#         finishLabel.configure(text="Download Complete")
#         title.configure(text=ytObject.title, text_color = "white")
#     except:
#         finishLabel.configure(text="YouTube link is invalid", text_color = "red")



# System Settings
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

# Our app frame
app = customtkinter.CTk()
app.geometry("720x480")
app.title("YouTube Downloader")

# Adding UI Elements
title = customtkinter.CTkLabel(app, text = "Insert a YouTube link:")
title.pack(padx=10, pady=10)

# Link input
url_var = tkinter.StringVar()
link = customtkinter.CTkEntry(app,width = 350, height = 40, textvariable = url_var) #entry is like an input
link.pack(padx=10, pady=10)

# Finished downloading
finishLabel = customtkinter.CTkLabel(app, text = "")
finishLabel.pack(padx=10, pady=10)

# Download button
download = customtkinter.CTkButton(app, text = "Download", command = start_download)
download.pack(padx=10, pady=10)



# Run app
app.mainloop()