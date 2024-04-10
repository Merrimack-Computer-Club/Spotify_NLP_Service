import pandas as pd
import matplotlib.pyplot as plt
import random
import base64
import io

'''
Constructs a Graph Image based off of the given data
Then encodes the image in base64 to be sent back to the server.
'''
def construct_Song_Emotions_Graph(data):
    # Plot size
    fig = plt.figure(figsize=(8, 6))
    #print(data)

    # Group by emotion and summarize
    counts = data.groupby("emotion").size().reset_index(name="count")
    # Reorder high to low
    counts = counts.sort_values("count", ascending=False)
    # Get most common emotion
    highest_emotion = counts.iloc[0]["emotion"]

    # Define style
    plt.style.use("seaborn")
    # Create and color bar graph bars
    bars = plt.barh(
        counts["emotion"], counts["count"], color="#1DB954", height=0.75
    )  # Define default color and width
    # For in range of bars, if highest emotion, color charcoal
    for i, bar in enumerate(bars):
        if counts.iloc[i]["emotion"] == highest_emotion:
            bar.set_color("#101820FF")
            bar.set_alpha(0.9)  # Set opacity
        # else, color spotify green
        else:
            bar.set_color("#1DB954")
            bar.set_alpha(0.7)  # Set opacity

    # Add plot margins
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.16)

    # Coor_flip
    plt.gca().invert_yaxis()

    # Background color of panel to match JS script
    plt.gcf().set_facecolor("#F5F5F5")

    # Labels -> define padding, font, and style
    plt.ylabel("Types of Emotion", labelpad=30, fontsize=10, fontweight="bold")
    plt.xlabel("Number of Occurences", labelpad=17, fontsize=10, fontweight="bold")
    plt.title("Sentimental Analysis", pad=15, fontsize=13, fontweight="bold")

    # Edit tick marks -> style, color, length, width, and padding
    plt.tick_params(axis="x", which="major", length=5, width=1, color="#D4D4D4", pad=10)
    plt.tick_params(axis="y", which="major", length=5, width=1, color="#D4D4D4", pad=10)

    # Add caption to bottom right of plot panel
    txt = "Data collected from Spotify API"
    plt.figtext(
        0.89, 0.01, txt, wrap=True, horizontalalignment="center", fontsize=8, style="italic"
    )

    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    encoded = base64.b64encode(my_stringIObytes.read()).decode()

    imgdata = base64.b64decode(encoded)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)

    return encoded

    # Save the graph

#construct_Song_Emotions_Graph(data)
