# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def sleeping_beauty(dataset, config, model, greed=False, chars_to_generate=30, text="Sleeping beauty is", temperature=1):
    with torch.no_grad():
        beauty = dataset.convert_from_string(text)
        # beauty = torch.cuda.FloatTensor(beauty).unsqueeze(0)
        beauty = torch.cuda.LongTensor(beauty).unsqueeze(0)
        # beauty = torch.tensor(beauty, dtype=torch.long, device=device).unsqueeze(0)
        one_hot = torch.cuda.FloatTensor(beauty.size(0),
                                         beauty.size(1),
                                         config.vocab_size).zero_()
        one_hot.scatter_(2, beauty.unsqueeze(-1), 1)

        sentence = []
        last_input = one_hot
        hidden = None

        for i in range(chars_to_generate + 1):
            out, hidden = model.forward(last_input, hidden)
            out = out[-1,:,:].unsqueeze(0)
            # print(hidden[0].shape)
            # print(hidden)
            # print(bh)
            hidden = (hidden[0][:,-1,:].unsqueeze(1).contiguous(), hidden[1][:,-1,:].unsqueeze(1).contiguous())
            # print(hidden[0].shape)
            # print(hidden)

            if greed:
                index = out.argmax().unsqueeze(-1)
            else:
                index = torch.multinomial(input=torch.softmax(out.squeeze(), dim=0)/temperature, num_samples=1)

            # print(index, index.shape)
            letter = torch.cuda.FloatTensor(1,1,config.vocab_size).zero_()
            letter.scatter_(2, index.unsqueeze(-1).unsqueeze(-1), 1)

            last_input = letter

            # print(sentence)
            sentence.append(letter.view(-1).argmax().item())
        # print(one_hot.shape)
        print("{} ... {}".format(text, dataset.convert_to_string(sentence)))

def random_int_sentence(dataset, config, model, greed=False, chars_to_generate=30, temperature=1):
    with torch.no_grad():
        # Generate some sentences by sampling from the model
        # why is generating a single random integer in pytorch so verbose
        rando = torch.randint(high=config.vocab_size, size=(1,1), dtype=torch.long, device=config.device)
        # rando2 = torch.randint(high=dataset.vocab_size, size=(1,1), dtype=torch.long, device=config.device)
        random_one_hot = torch.cuda.FloatTensor(1,1,config.vocab_size).zero_()
        random_one_hot.scatter_(2, rando.unsqueeze(-1), 1)
        sentence = []
        sentence.append(random_one_hot.view(-1).argmax().item())
        last_input = random_one_hot
        hidden = None
        for i in range(chars_to_generate):
            out, hidden = model.forward(last_input, hidden)
            # The output is NOT a one-hot vector. Usually we simply take the
            # argmax (greedy) as the output but instead we sample randomly to
            # determine the letter
            # print(index.shape, index)
            if greed:
                index = out.argmax().unsqueeze(-1)
            else:
                # index = torch.multinomial(input=out.squeeze(), num_samples=1)
                # print(index, out.squeeze(), torch.softmax(out.squeeze(), dim=0))
                index = torch.multinomial(input=torch.softmax(out.squeeze(), dim=0)/temperature, num_samples=1)
                # print(asdasd)
            # print(index.shape, index, out.shape)
            # print(index.shape, index.unsqueeze(-1).unsqueeze(-1).shape, rando.unsqueeze(-1).shape)
            letter = torch.cuda.FloatTensor(1,1,config.vocab_size).zero_()
            letter.scatter_(2, index.unsqueeze(-1).unsqueeze(-1), 1)

            last_input = letter

            # print(sentence)
            sentence.append(letter.view(-1).argmax().item())

    print(dataset.convert_to_string(sentence))

def train(config):

    # Initialize the device which to run the model on
    config.device = 'cuda'
    device = torch.device(config.device)

    # Initialize the model that we are going to use


    dataset = TextDataset(config.txt_file, config.seq_length)

    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    vocab_size = dataset.vocab_size
    config.vocab_size = vocab_size

    model = TextGenerationModel(config.batch_size, config.seq_length, vocab_size,
                 config.lstm_num_hidden, config.lstm_num_layers, config.device)
    model = model.to(device)
    # Initialize the dataset and data loader (note the +1)


    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    for epoch in range(100):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################


            # print(batch_inputs)
            # print(asdasd)
            batch_inputs = torch.stack(batch_inputs).to(device)
            # print(batch_inputs.shape)
            # batch_inputs = F.one_hot(batch_inputs, vocab_size)

            one_hot = torch.cuda.FloatTensor(batch_inputs.size(0),
                                             batch_inputs.size(1),
                                             vocab_size).zero_()
            one_hot.scatter_(2, batch_inputs.unsqueeze(-1), 1)

            # make batch first dim
            batch_targets = torch.stack(batch_targets, dim = 1).to(device)

            # one_hot_targets = torch.cuda.FloatTensor(batch_inputs.size(0),
            #                                  batch_inputs.size(1),
            #                                  vocab_size).zero_()
            # one_hot_targets.scatter_(2, batch_targets.unsqueeze(-1), 1)
            # print(one_hot.shape)
            # print(one_hot[0,0,:])
            # print(batch_targets.shape, batch_inputs.shape)
            # print(batch_inputs, batch_targets)
            out, _ = model.forward(one_hot)

            # print(out.argmax(2).shape, batch_targets.shape)
            # print(asdasd)
            # The data is (sequence,batch,one-hot) (30, 64, 87)
            # but criterion gets angry, you can keep the batch targets as index
            # but the input must be the shape (sequence, one-hot, batch)?

            # loss = criterion(out.argmax(2), batch_targets)
            # print(out.shape, batch_targets.shape)
            # print(out.transpose(2,1).shape)
            # print(asdasd)
            # all these errors yelling at me
            loss = criterion(out.transpose(2,1), batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % config.print_every == 0:

                # print(out.shape, batch_targets.shape)
                compare = (out.argmax(2) == batch_targets)
                # print(compare.shape)

                summed = compare.sum().item()
                # print(compare.numel())
                # print(asdasd)
                accuracy = summed/compare.numel()
                print("[{}] Train Step {:04d}/{:d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), int(step),
                        int(config.train_steps), config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if step % config.sample_every == 0:
                sleeping_beauty(dataset, config, model)
                # random_int_sentence(dataset, config, model)


            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
