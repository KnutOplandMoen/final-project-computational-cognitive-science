"""
Listener module for the model. Defines the behavior and decision-making process of the listener agents based on the observed information and their beliefs.

Should import priors, stories etc from the other modules and be a simple implementation in memo. 
"""

import memo
from model.prior import prior
from model.fit import Record, Story, pi

@memo
def listener[note: Story, h: Record, a: Story](rho, alpha):
    """
    Computes the probability that the listener adopts a given story `a` based on the observed note `note` and record `h`, using the model parameters `rho` and `alpha`.
    
    Args:
        note::Story
            The observed note of type `Story`.
        h::Record
            The observed record of type `Record`.
        a::Story
            The story the listener may adopt of type `Story`.
        rho::float
            Model parameter for the prior probability.
        alpha::float
            Model parameter for the listener's decision-making.
    Returns:
        a::float
            Probability that the listener adopts story `a`.

    """ 
    
    listener: knows(note, h)  
    listener: thinks[
        world: knows(note),
        world: chooses(story in Story, wpp=prior(story, note, rho)),   
        world: chooses(rec in Record, wpp=pi(story, rec)),             
    ]
    listener: observes [world.rec] is h                                
    listener: chooses(adopt in Story, wpp=Pr[world.story == adopt] ** alpha)
    return Pr[listener.adopt == a]

