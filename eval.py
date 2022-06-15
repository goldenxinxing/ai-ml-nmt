import os
import random
import torch
from config import EOS_token, SOS_token
from dataset import prepareData
from helper import MAX_LENGTH, tensorFromSentence
from model import AttnDecoderRNN, EncoderRNN
from nltk.metrics import *
from nltk.translate import *
from nltk.translate.bleu_score import SmoothingFunction


_ROOT_DIR = os.path.dirname(__file__)
_ENCODER_MODEL_PATH = os.path.join(_ROOT_DIR, "models/encoder.pth")
_DECODER_MODEL_PATH = os.path.join(_ROOT_DIR, "models/decoder.pth")
_EVAL_RESULT_PATH = os.path.join(_ROOT_DIR, "results/pred.txt")
_EVAL_LABEL_PATH = os.path.join(_ROOT_DIR, "results/label.txt")

def evaluate(device, input_lang, output_lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                #decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = bleu([[ref[1:-1]] for ref in references],
                      [hyp[1:-1] for hyp in hypotheses])

    return bleu_score


def evaluateRandomly(device, input_lang, output_lang, pairs, encoder, decoder, n=10):
    #input_sent = []
    #output_sent = []
    #pred_sent = []
    for i in range(n):
        output_sent = []
        pred_sent = []
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_sent.append(pair[1])

        append_new_line(_EVAL_LABEL_PATH, pair[1])

        output_words, attentions = evaluate(device, input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)

        pred_sent.append(output_sentence)
        append_new_line(_EVAL_RESULT_PATH, output_sentence)

        # valid_metric = bleu(pair[1].split(), output_sentence.split())
        print('score', bleu([pair[1].split()], output_sentence.split(), smoothing_function=SmoothingFunction().method2))
        # print('accuracy', accuracy(pair[1].split(), output_sentence.split()))
        print('')
    # bleu scroe
    # data = zip(input_sent, output_sent)

    # valid_metric = get_bleu([tgt for src, tgt in data], pred_sent)
    # print('score', valid_metric)
    
    #print('score-my0', bleu("i am gxx".split(), "i am gxx".split(), smoothing_function=SmoothingFunction().method0))
    print('score-my1', bleu(["i am gxx o o".split()], "i am gxx o o".split(), smoothing_function=SmoothingFunction().method1))
    print('score-my2', bleu(["i am gxx o o".split()], "i am gxx o o".split(), smoothing_function=SmoothingFunction().method2))
    print('score-my3', bleu(["i am gxx o o".split()], "i am gxx o o".split(), smoothing_function=SmoothingFunction().method3))
    print('score-my4', bleu(["i am gxx o o".split()], "i am gxx o o".split(), smoothing_function=SmoothingFunction().method4))

    
    #print('score-my0', bleu("i am gxx ii asdf fdf fd df f".split(), "i am y gxx ii asdf y u fdf fd df f u".split(), smoothing_function=SmoothingFunction().method0))
    print('score-my1', bleu(["i am gxx ii asdf fdf fd df f".split()], "i am gxx ii asdf fdf fd df f".split(), smoothing_function=SmoothingFunction().method1))
    print('score-my2', bleu(["i am gxx ii asdf fdf fd df f".split()], "i am gxx ii asdf fdf fd df f".split(), smoothing_function=SmoothingFunction().method2))
    print('score-my3', bleu(["i am gxx ii asdf fdf fd df f".split()], "i am gxx ii asdf fdf fd df f".split(), smoothing_function=SmoothingFunction().method3))
    print('score-my4', bleu(["i am gxx ii asdf fdf fd df f".split()], "i am gxx ii asdf fdf fd df f".split(), smoothing_function=SmoothingFunction().method4))
    
    print('score-my1', bleu(["je te donnerai ce livre".split()], "je te donne ce livre".split(), smoothing_function=SmoothingFunction().method1))
    print('score-my2', bleu(["je te donnerai ce livre".split()], "je te donne ce livre".split(), smoothing_function=SmoothingFunction().method2))
    print('score-my3', bleu(["je te donnerai ce livre".split()], "je te donne ce livre".split(), smoothing_function=SmoothingFunction().method3))
    print('score-my4', bleu(["je te donnerai ce livre".split()], "je te donne ce livre".split(), smoothing_function=SmoothingFunction().method4))


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = prepareData('eng', 'fra', False)
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)

    net1 = torch.load(_ENCODER_MODEL_PATH, device)
    encoder1.load_state_dict(net1)


    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, device, dropout_p=0.1).to(device)

    net2 = torch.load(_DECODER_MODEL_PATH, device)
    attn_decoder1.load_state_dict(net2)

    evaluateRandomly(device, input_lang, output_lang, pairs, encoder1, attn_decoder1)