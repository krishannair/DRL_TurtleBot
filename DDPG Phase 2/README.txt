The models saved without description are with 
Rcol = -1000
Rgoal = 1000
R if dist>0 : 500*dist + 3*angle
  else R : -10

Next change was addition of Ravoidance = -100*avoidance_rate
avoidance rate first was 0.29 then 0.14 and then 0.09

Next change is for dist<= 0
R = 500*dist + -100*av rate

Have to add decaying epsilon

Now making neg_dist have fixed high -ve reward and angle reward negative if heading is 
greater than pi/2 also the avoidance penalty has increased and made 10 times more so as 
to make it comparable to distance reward. Angle reward is from -4.5(-3pi/2) to 4.5 now.
So now RF = 500*dist + 3*angle - 1000* avoidance
also the max time the bot has to reach next goal has been incereased from 25 to 40. 

Now next trying to increase avoidance to make it lesser likely to collide.
Maybe trying 2000 as the avoidance factor.

In 2000 it is afraid to move towards obstacles and moves back and forth a lot
not reaching the goal in the given 40 seconds usually, 
now checking if 1500 as the avoidance factor will be good.

1500 turns out to have the worst of both it moves back and forth as well but also does collide.

Trying a high number of episodes with 4 obs something like 4000 episodes.

Trying 4000 episodes we can see that the robot doesn't improve its score after about 1000 episodes. It is trying 
to go around the obstacles more than reaching the goal. Therefore reducing avoidance factor to 500 and trying 1000 
episodes in 4 obstacle environment. Also checked how big the noise is compared to the action being performed and 
it turns out that the noise is sometimes as big as the maximum action that can be performed so now made sigma of the
OUNoise from 0.15 to 0.05 to make noise from around 0.2 to 0.066.

Now final scores were not very high so now trying to give avoidance rate as a fixed 0.03 and avoidance factor as 1000
to see if that helps the bot avoid the obstacles because it still does collide sometimes after 1000 episodes. Also changed 
the sigma to a lower value of 0.025 to see how it affects the scores. Will make it higher again if unsatisfactory output.

Changing avoidance rate to a fixed value seems to make navigation very difficult for the bot as it doesn't want to go anywhere
near the obstacle and has no other path so finally gives up. Hence changing the Avoidance rate back to being dynamic and also 
increasing the reward for reaching the goal to 2000. 
Also Turns out there was a bug in the code and the penalty for not reaching next goal within 40s after reaching a goal 
was not being conveyed to the RL model and only being reflected in the plot. Now giving -250 penalty for doing so.
