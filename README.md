Central hub for dynamic auto adjustments of hyper parameters in my transformer.. and meta learner( that parts a wip ). Right now it just works with base frequency (rope) 
and window size of the hybrid attention block based on loss during training.. 
It's overzealous atm (turned up to 11).

Some of the blocks are not connected yet in this version of the model but the dynamic feedback part is which auto adjusts the window size and base frequency. Fully working model just not all the ideas here are connected yet.. but they are there. Full script with hugging face trainer and dataset integration.. Only did it because we have to these days. Otherwise works great/better with a pytorch loop. 
