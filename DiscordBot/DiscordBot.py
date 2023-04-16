import torch
from discord.ext import commands
import discord
import DiscordBotFunctions as db_functions
import matplotlib.pyplot as plt

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='/', intents=intents)

base_line_encoding = None
@bot.event
async def on_ready():
    print("Ready")
    pass

@bot.command(name="SetBaseLine")
async def set_base_line(ctx):
    url = ctx.message.attachments[0].url
    image_file = db_functions.get_img_from_url(url)
    image_tensor = db_functions.convert_image_file_to_tensor(image_file=image_file)
    permuted_image = (torch.permute(image_tensor, (1,2,0))*255).int()
    permuted_image = permuted_image[:, :, :3]
    cropped_image = db_functions.detect_face_in_image(image_tensor=permuted_image)
    if cropped_image == None:
        await ctx.send("No face was found in the picture. Set another baseline")
        return
    db_functions.plot_image(torch.permute(cropped_image, (1,2,0)))
    global base_line_encoding
    base_line_encoding = db_functions.encode_face(cropped_image=cropped_image)
    pass

@bot.command(name="Verify")
async def verify_face_against_base_line(ctx):
    if base_line_encoding == None:
        await ctx.send("Baseline image hasn't been set for this image to be compared to")
        return
    url = ctx.message.attachments[0].url
    image_file = db_functions.get_img_from_url(url)
    image_tensor = db_functions.convert_image_file_to_tensor(image_file=image_file)
    permuted_image = (torch.permute(image_tensor, (1, 2, 0)) * 255).int()
    permuted_image = permuted_image[:,:,:3]
    cropped_image = db_functions.detect_face_in_image(image_tensor=permuted_image)
    if cropped_image == None:
        await ctx.send("No face was found in the picture")
        return
    db_functions.plot_image(torch.permute(cropped_image, (1, 2, 0)))
    unknown_encoding = db_functions.encode_face(cropped_image=cropped_image)
    distance, classification = db_functions.get_verification_determination(base_line_encoding, unknown_encoding)
    await ctx.send("Distance = {0} ".format(distance))
    await ctx.send("Classification Probability = {0}".format(str(torch.squeeze(classification).item())))

Token = "TOKEN"

bot.run(token=Token)