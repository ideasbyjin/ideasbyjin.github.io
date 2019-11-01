---
layout: post
title: "Supervised learning demo: what position do I play?"
date: 2019-11-01
category: stats
mathjax: true
---

Last time I covered a section on clustering, a group of _unsupervised_ learning methods – so called because
they are not given the class memberships of the data$$^\dagger$$. Don't worry, I will do more posts on
clustering soon. For now I wanted to give a quick overview of what _supervised_ methods look like. For that, let's
look at the statistics of hockey players!

$$\dagger$$: this is a gross generalisation. More formally, for some dataset $$\mathbf{X}$$, if we are trying
to predict an output variable $$\mathbf{Y}$$, we use supervised learning methods, otherwise unsupervised
learning methods.

Hockey (on ice, obviously!) is a game where we have 5 skaters and 1 goalie per rotation.
The 5 skaters can be divided into three$$^\ddagger$$ subclasses:
* The wingers (LW, RW)
* The centre (C)
* Defensemen (D)

$$\ddagger$$: the centre and wingers can also be bundled up as forwards.

<div style="text-align: center">
    <img src="/assets/notebooks/hockey/hockey_players_position.png"/>
</div>

Typically, each skater's actions are recorded, which include:
* Goal(s) scored, assist(s) made [i.e. did the player make a pass leading up to the goal?]
* Penalties in minutes (infringements typically lead to 2 minute bans from the game)
* Ice time in 5-on-5 or 'penalty kill' situations
* Shots blocked
etc.

Usually by looking at these statistics, one can have an approximate idea of the position a given hockey player plays. To many, this might seem like a pretty easy problem. Surely forwards are supposed to score goals! Defensemen are supposed to block shots!

#### Wait, so why do you want to know a player's position?
Predicting a player's position is perhaps not the first classification that comes to mind. However, it's useful for something like fantasy sports leagues. In fantasy sports, you typically have roster slots for _n_ centres, _m_ defensemen, and _q_ wingers. Using those constraints, you want to (usually) maximise every statistical category.

For example, if a fantasy team has a bunch of goal-scoring centremen, which position player do we pick out next to max out the penalties in minutes category? Simultaneously, which position also happens to maximise the number of assists? Do we pick up a defenseman, or a gritty right wing?

Hockey is slightly more complex than meets the eye. Typically, being a defenseman or forward can largely constrain your statistical profile; there are some defensemen that are very talented on offense (e.g. Morgan Reilly), and some forwards who are tougher, and deployed on a "checking" line to provide strength.

![MR](https://imagesvc.timeincapp.com/v3/fan/image?url=https%3A%2F%2Feditorinleaf.com%2Fwp-content%2Fuploads%2Fgetty-images%2F2018%2F06%2F949930044.jpeg&c=sc&w=736&h=485)

(This is Morgan Reilly.)

So, how can we predict positions using stats? Let's find out.

**If you have 30 seconds...**
* This post is really a demo of various supervised learning methods. See table below for a quick overview.
* The algorithm of choice is often dependent on use case, and you should consider questions like:
    * What is the distribution of your output?
    * How can we interpret the model?
    
**If you have 10 minutes...**
* Read on. I've tried to section this entry based on what bits you might be interested in. This post is very much intended to be a whirlwind tour of the various supervised learning methods, rather than a deep-dive.


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```

### Data cleanup


```python
# Read in the data from Kaggle
df = pd.read_csv("game_skater_stats.csv")

# We'll use this later.
pinfo = pd.read_csv("player_info.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>player_id</th>
      <th>team_id</th>
      <th>timeOnIce</th>
      <th>assists</th>
      <th>goals</th>
      <th>shots</th>
      <th>hits</th>
      <th>powerPlayGoals</th>
      <th>powerPlayAssists</th>
      <th>...</th>
      <th>faceoffTaken</th>
      <th>takeaways</th>
      <th>giveaways</th>
      <th>shortHandedGoals</th>
      <th>shortHandedAssists</th>
      <th>blocked</th>
      <th>plusMinus</th>
      <th>evenTimeOnIce</th>
      <th>shortHandedTimeOnIce</th>
      <th>powerPlayTimeOnIce</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2011030221</td>
      <td>8467412</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>885</td>
      <td>98</td>
      <td>16</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2011030221</td>
      <td>8468501</td>
      <td>1</td>
      <td>1168</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>767</td>
      <td>401</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2011030221</td>
      <td>8470609</td>
      <td>1</td>
      <td>558</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>542</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2011030221</td>
      <td>8471816</td>
      <td>1</td>
      <td>1134</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>935</td>
      <td>183</td>
      <td>16</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2011030221</td>
      <td>8472410</td>
      <td>1</td>
      <td>436</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>436</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



The skater stats are given per player (`player_id`), and per game (`game_id`) that they have played. We know from some good documentation that:
* The first four digits represent the season (e.g. 2010-2011 season)
* The next two digits represent whether the game was held in the regular season or playoffs, etc.

What we will do is some clever `pandas` magic to:
* Only use regular season games
* Aggregate the statistics per player


```python
# Filter for regular season and annotate season ID
# https://github.com/dword4/nhlapi#game-ids

df['game_id'] = df['game_id'].astype(str)
reg_season = df[df['game_id'].apply(lambda x: x[4:6] == "02")].copy()
reg_season['Season'] = reg_season['game_id'].apply(lambda x: x[:4])
```


```python
reg_season.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_id</th>
      <th>player_id</th>
      <th>team_id</th>
      <th>timeOnIce</th>
      <th>assists</th>
      <th>goals</th>
      <th>shots</th>
      <th>hits</th>
      <th>powerPlayGoals</th>
      <th>powerPlayAssists</th>
      <th>...</th>
      <th>takeaways</th>
      <th>giveaways</th>
      <th>shortHandedGoals</th>
      <th>shortHandedAssists</th>
      <th>blocked</th>
      <th>plusMinus</th>
      <th>evenTimeOnIce</th>
      <th>shortHandedTimeOnIce</th>
      <th>powerPlayTimeOnIce</th>
      <th>Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25379</td>
      <td>2013020674</td>
      <td>8471817</td>
      <td>19</td>
      <td>547</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>547</td>
      <td>0</td>
      <td>0</td>
      <td>2013</td>
    </tr>
    <tr>
      <td>25380</td>
      <td>2013020674</td>
      <td>8467890</td>
      <td>19</td>
      <td>1103</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>999</td>
      <td>104</td>
      <td>0</td>
      <td>2013</td>
    </tr>
    <tr>
      <td>25381</td>
      <td>2013020674</td>
      <td>8475768</td>
      <td>19</td>
      <td>1119</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>955</td>
      <td>78</td>
      <td>86</td>
      <td>2013</td>
    </tr>
    <tr>
      <td>25382</td>
      <td>2013020674</td>
      <td>8473534</td>
      <td>19</td>
      <td>1006</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>930</td>
      <td>42</td>
      <td>34</td>
      <td>2013</td>
    </tr>
    <tr>
      <td>25383</td>
      <td>2013020674</td>
      <td>8466160</td>
      <td>19</td>
      <td>593</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>593</td>
      <td>0</td>
      <td>0</td>
      <td>2013</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# aggregate stats and use average for time; otherwise totals
# We could do this by season but we'll stick to overall totals for simplification.
aggregated_stats = reg_season.groupby('player_id').agg(
    {
        "game_id": len, # use this as an aggregating function to get number of games played
        "timeOnIce": "mean",
        "goals": "sum",
        "assists": "sum",
        "shots": "sum",
        "hits": "sum",
        "powerPlayGoals":"sum",
        "powerPlayAssists": "sum",
        "penaltyMinutes": "sum",
        "faceOffWins": "sum",
        "faceoffTaken": "sum",
        "takeaways": "sum",
        "giveaways": "sum",
        "shortHandedGoals": "sum",
        "shortHandedAssists": "sum",
        "blocked": "sum",
        "plusMinus": "sum",
        "evenTimeOnIce": "mean",
        "shortHandedTimeOnIce": "mean",
    }
)
aggregated_stats.columns = ['games_played'] + list(aggregated_stats.columns[1:])
aggregated_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>games_played</th>
      <th>timeOnIce</th>
      <th>goals</th>
      <th>assists</th>
      <th>shots</th>
      <th>hits</th>
      <th>powerPlayGoals</th>
      <th>powerPlayAssists</th>
      <th>penaltyMinutes</th>
      <th>faceOffWins</th>
      <th>faceoffTaken</th>
      <th>takeaways</th>
      <th>giveaways</th>
      <th>shortHandedGoals</th>
      <th>shortHandedAssists</th>
      <th>blocked</th>
      <th>plusMinus</th>
      <th>evenTimeOnIce</th>
      <th>shortHandedTimeOnIce</th>
    </tr>
    <tr>
      <th>player_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8446485</td>
      <td>47</td>
      <td>626.978723</td>
      <td>6</td>
      <td>5</td>
      <td>57</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>95</td>
      <td>168</td>
      <td>16</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>573.255319</td>
      <td>51.191489</td>
    </tr>
    <tr>
      <td>8448208</td>
      <td>460</td>
      <td>1039.756522</td>
      <td>120</td>
      <td>202</td>
      <td>1041</td>
      <td>114</td>
      <td>36</td>
      <td>48</td>
      <td>260</td>
      <td>2</td>
      <td>8</td>
      <td>199</td>
      <td>276</td>
      <td>0</td>
      <td>0</td>
      <td>73</td>
      <td>47</td>
      <td>872.436957</td>
      <td>1.308696</td>
    </tr>
    <tr>
      <td>8449645</td>
      <td>40</td>
      <td>746.900000</td>
      <td>4</td>
      <td>11</td>
      <td>79</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>8</td>
      <td>133</td>
      <td>274</td>
      <td>14</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>-4</td>
      <td>594.250000</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <td>8450725</td>
      <td>81</td>
      <td>966.283951</td>
      <td>14</td>
      <td>34</td>
      <td>132</td>
      <td>81</td>
      <td>6</td>
      <td>11</td>
      <td>35</td>
      <td>52</td>
      <td>97</td>
      <td>16</td>
      <td>32</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>13</td>
      <td>761.962963</td>
      <td>43.308642</td>
    </tr>
    <tr>
      <td>8455919</td>
      <td>18</td>
      <td>558.111111</td>
      <td>2</td>
      <td>0</td>
      <td>11</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>101</td>
      <td>189</td>
      <td>4</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>-1</td>
      <td>414.444444</td>
      <td>140.055556</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8481442</td>
      <td>1</td>
      <td>760.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>760.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8481477</td>
      <td>2</td>
      <td>767.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>717.000000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <td>8481479</td>
      <td>2</td>
      <td>801.000000</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>-1</td>
      <td>801.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8481481</td>
      <td>1</td>
      <td>672.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>672.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8481486</td>
      <td>1</td>
      <td>1263.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1152.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
<p>1995 rows × 19 columns</p>
</div>



Let's do some more feature engineering to make our lives easier...


```python
# Powerplay and shorthanded goals/assists are typically much lower than regular goals/assists, so it's convenient to take the sum.
# Faceoffs are typically reported in percentages, anyway.
aggregated_stats['powerPlayPoints'] = aggregated_stats['powerPlayGoals'] + aggregated_stats['powerPlayAssists']
aggregated_stats['shortHandedPoints'] = aggregated_stats['shortHandedGoals'] + aggregated_stats['shortHandedAssists']

# Since some players never take faceOffs, just stick to 0 to avoid zero division errors
percentage = (aggregated_stats['faceOffWins'] / aggregated_stats['faceoffTaken'])*100
percentage = [ _ if not np.isnan(_) else 0 for _ in percentage ]

aggregated_stats['faceOffPercentage'] = percentage
aggregated_stats.drop(columns=['powerPlayGoals', 'powerPlayAssists', 'shortHandedGoals', 'shortHandedAssists', 'faceOffWins', 'faceoffTaken'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>games_played</th>
      <th>timeOnIce</th>
      <th>goals</th>
      <th>assists</th>
      <th>shots</th>
      <th>hits</th>
      <th>penaltyMinutes</th>
      <th>takeaways</th>
      <th>giveaways</th>
      <th>blocked</th>
      <th>plusMinus</th>
      <th>evenTimeOnIce</th>
      <th>shortHandedTimeOnIce</th>
      <th>powerPlayPoints</th>
      <th>shortHandedPoints</th>
      <th>faceOffPercentage</th>
    </tr>
    <tr>
      <th>player_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8446485</td>
      <td>47</td>
      <td>626.978723</td>
      <td>6</td>
      <td>5</td>
      <td>57</td>
      <td>45</td>
      <td>12</td>
      <td>16</td>
      <td>8</td>
      <td>11</td>
      <td>1</td>
      <td>573.255319</td>
      <td>51.191489</td>
      <td>0</td>
      <td>0</td>
      <td>56.547619</td>
    </tr>
    <tr>
      <td>8448208</td>
      <td>460</td>
      <td>1039.756522</td>
      <td>120</td>
      <td>202</td>
      <td>1041</td>
      <td>114</td>
      <td>260</td>
      <td>199</td>
      <td>276</td>
      <td>73</td>
      <td>47</td>
      <td>872.436957</td>
      <td>1.308696</td>
      <td>84</td>
      <td>0</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <td>8449645</td>
      <td>40</td>
      <td>746.900000</td>
      <td>4</td>
      <td>11</td>
      <td>79</td>
      <td>7</td>
      <td>8</td>
      <td>14</td>
      <td>13</td>
      <td>10</td>
      <td>-4</td>
      <td>594.250000</td>
      <td>0.575000</td>
      <td>6</td>
      <td>0</td>
      <td>48.540146</td>
    </tr>
    <tr>
      <td>8450725</td>
      <td>81</td>
      <td>966.283951</td>
      <td>14</td>
      <td>34</td>
      <td>132</td>
      <td>81</td>
      <td>35</td>
      <td>16</td>
      <td>32</td>
      <td>25</td>
      <td>13</td>
      <td>761.962963</td>
      <td>43.308642</td>
      <td>17</td>
      <td>1</td>
      <td>53.608247</td>
    </tr>
    <tr>
      <td>8455919</td>
      <td>18</td>
      <td>558.111111</td>
      <td>2</td>
      <td>0</td>
      <td>11</td>
      <td>21</td>
      <td>8</td>
      <td>4</td>
      <td>10</td>
      <td>10</td>
      <td>-1</td>
      <td>414.444444</td>
      <td>140.055556</td>
      <td>0</td>
      <td>0</td>
      <td>53.439153</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8481442</td>
      <td>1</td>
      <td>760.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>760.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8481477</td>
      <td>2</td>
      <td>767.000000</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>717.000000</td>
      <td>43.000000</td>
      <td>0</td>
      <td>0</td>
      <td>29.411765</td>
    </tr>
    <tr>
      <td>8481479</td>
      <td>2</td>
      <td>801.000000</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>-1</td>
      <td>801.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8481481</td>
      <td>1</td>
      <td>672.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>672.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <td>8481486</td>
      <td>1</td>
      <td>1263.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1152.000000</td>
      <td>2.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1995 rows × 16 columns</p>
</div>



Finally, for each player, if they have played fewer than 41 games, let's remove them. I chose the number 41 because
there are 82 games in a season. I want to know that a player has played at least half a season's worth of games, otherwise we would have very little data to work with.


```python
sufficient_games = []
for n,g in aggregated_stats.groupby('player_id'):
    if g['games_played'].sum() >= 41:
        sufficient_games.append(n)

final_stats = aggregated_stats[aggregated_stats.index.get_level_values("player_id").isin(sufficient_games)].copy()
final_stats_players = final_stats.index.get_level_values('player_id')
final_stats
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>games_played</th>
      <th>timeOnIce</th>
      <th>goals</th>
      <th>assists</th>
      <th>shots</th>
      <th>hits</th>
      <th>powerPlayGoals</th>
      <th>powerPlayAssists</th>
      <th>penaltyMinutes</th>
      <th>faceOffWins</th>
      <th>...</th>
      <th>giveaways</th>
      <th>shortHandedGoals</th>
      <th>shortHandedAssists</th>
      <th>blocked</th>
      <th>plusMinus</th>
      <th>evenTimeOnIce</th>
      <th>shortHandedTimeOnIce</th>
      <th>powerPlayPoints</th>
      <th>shortHandedPoints</th>
      <th>faceOffPercentage</th>
    </tr>
    <tr>
      <th>player_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8446485</td>
      <td>47</td>
      <td>626.978723</td>
      <td>6</td>
      <td>5</td>
      <td>57</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>95</td>
      <td>...</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>573.255319</td>
      <td>51.191489</td>
      <td>0</td>
      <td>0</td>
      <td>56.547619</td>
    </tr>
    <tr>
      <td>8448208</td>
      <td>460</td>
      <td>1039.756522</td>
      <td>120</td>
      <td>202</td>
      <td>1041</td>
      <td>114</td>
      <td>36</td>
      <td>48</td>
      <td>260</td>
      <td>2</td>
      <td>...</td>
      <td>276</td>
      <td>0</td>
      <td>0</td>
      <td>73</td>
      <td>47</td>
      <td>872.436957</td>
      <td>1.308696</td>
      <td>84</td>
      <td>0</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <td>8450725</td>
      <td>81</td>
      <td>966.283951</td>
      <td>14</td>
      <td>34</td>
      <td>132</td>
      <td>81</td>
      <td>6</td>
      <td>11</td>
      <td>35</td>
      <td>52</td>
      <td>...</td>
      <td>32</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>13</td>
      <td>761.962963</td>
      <td>43.308642</td>
      <td>17</td>
      <td>1</td>
      <td>53.608247</td>
    </tr>
    <tr>
      <td>8456283</td>
      <td>47</td>
      <td>1075.510638</td>
      <td>0</td>
      <td>8</td>
      <td>23</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>-9</td>
      <td>896.234043</td>
      <td>170.702128</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8457063</td>
      <td>152</td>
      <td>1416.460526</td>
      <td>27</td>
      <td>69</td>
      <td>323</td>
      <td>90</td>
      <td>11</td>
      <td>45</td>
      <td>48</td>
      <td>0</td>
      <td>...</td>
      <td>56</td>
      <td>0</td>
      <td>1</td>
      <td>171</td>
      <td>19</td>
      <td>1049.921053</td>
      <td>133.348684</td>
      <td>56</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8480839</td>
      <td>82</td>
      <td>1269.329268</td>
      <td>9</td>
      <td>35</td>
      <td>177</td>
      <td>97</td>
      <td>5</td>
      <td>15</td>
      <td>34</td>
      <td>0</td>
      <td>...</td>
      <td>79</td>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>-13</td>
      <td>1088.743902</td>
      <td>9.500000</td>
      <td>20</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8480943</td>
      <td>53</td>
      <td>840.905660</td>
      <td>3</td>
      <td>4</td>
      <td>43</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>...</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>0</td>
      <td>835.075472</td>
      <td>4.113208</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8480944</td>
      <td>65</td>
      <td>680.800000</td>
      <td>1</td>
      <td>12</td>
      <td>65</td>
      <td>58</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>204</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>55</td>
      <td>5</td>
      <td>581.415385</td>
      <td>95.230769</td>
      <td>0</td>
      <td>0</td>
      <td>50.122850</td>
    </tr>
    <tr>
      <td>8480946</td>
      <td>82</td>
      <td>849.024390</td>
      <td>13</td>
      <td>24</td>
      <td>137</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>22</td>
      <td>...</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>10</td>
      <td>781.475610</td>
      <td>3.963415</td>
      <td>2</td>
      <td>0</td>
      <td>36.065574</td>
    </tr>
    <tr>
      <td>8480950</td>
      <td>41</td>
      <td>824.365854</td>
      <td>0</td>
      <td>4</td>
      <td>27</td>
      <td>150</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>61</td>
      <td>-9</td>
      <td>807.512195</td>
      <td>14.463415</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1367 rows × 22 columns</p>
</div>



## Train, validate, test

In machine learning, training, validating, and testing your model is a fundamental piece of the puzzle. Without proper splits of your data, there is a potential to overfit your model to the training set. Furthermore, the split datasets should have similar distributions of classes so that you avoid overfitting/over-penalisation, too. For the sake of this blog, I will only split by training and testing, and make one split.
There are other split strategies like $$k$$-fold cross-validation but... we won't talk about that for now. Back to topic!

Splitting is best done using `sklearn`'s builtin train-test splitter:


```python
from sklearn.model_selection import train_test_split

# Get skater data from pinfo
skaters = pinfo[pinfo['primaryPosition'] != "G"][['player_id', 'firstName', 'lastName', 'primaryPosition']].copy()
skaters = skaters[skaters['player_id'].isin(final_stats_players)]

# the stratify argument makes sure we split our dataset
# so that even though the test set is 1/3 the size of the training set
# it has a similar distribution of wingers, defensemen... etc.
# let's use a seed of 0.
training_ids, test_ids = train_test_split(skaters['player_id'], random_state = 0,
                                               test_size = 0.25, stratify = skaters['primaryPosition'])
```


```python
# get the training set of data.
# Since aggregated_stats is aggregated on both player id and season,
# we have a multi-index object. this is a way to search on one column of that index.
playerIdIndex = aggregated_stats.index.get_level_values("player_id")

# Get the training set and test set of data.
training_set = aggregated_stats[playerIdIndex.isin(training_ids)].copy()
test_set = aggregated_stats[playerIdIndex.isin(test_ids)].copy()
training_set.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>games_played</th>
      <th>timeOnIce</th>
      <th>goals</th>
      <th>assists</th>
      <th>shots</th>
      <th>hits</th>
      <th>powerPlayGoals</th>
      <th>powerPlayAssists</th>
      <th>penaltyMinutes</th>
      <th>faceOffWins</th>
      <th>...</th>
      <th>giveaways</th>
      <th>shortHandedGoals</th>
      <th>shortHandedAssists</th>
      <th>blocked</th>
      <th>plusMinus</th>
      <th>evenTimeOnIce</th>
      <th>shortHandedTimeOnIce</th>
      <th>powerPlayPoints</th>
      <th>shortHandedPoints</th>
      <th>faceOffPercentage</th>
    </tr>
    <tr>
      <th>player_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8448208</td>
      <td>460</td>
      <td>1039.756522</td>
      <td>120</td>
      <td>202</td>
      <td>1041</td>
      <td>114</td>
      <td>36</td>
      <td>48</td>
      <td>260</td>
      <td>2</td>
      <td>...</td>
      <td>276</td>
      <td>0</td>
      <td>0</td>
      <td>73</td>
      <td>47</td>
      <td>872.436957</td>
      <td>1.308696</td>
      <td>84</td>
      <td>0</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <td>8450725</td>
      <td>81</td>
      <td>966.283951</td>
      <td>14</td>
      <td>34</td>
      <td>132</td>
      <td>81</td>
      <td>6</td>
      <td>11</td>
      <td>35</td>
      <td>52</td>
      <td>...</td>
      <td>32</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
      <td>13</td>
      <td>761.962963</td>
      <td>43.308642</td>
      <td>17</td>
      <td>1</td>
      <td>53.608247</td>
    </tr>
    <tr>
      <td>8456283</td>
      <td>47</td>
      <td>1075.510638</td>
      <td>0</td>
      <td>8</td>
      <td>23</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>0</td>
      <td>...</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>-9</td>
      <td>896.234043</td>
      <td>170.702128</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8457063</td>
      <td>152</td>
      <td>1416.460526</td>
      <td>27</td>
      <td>69</td>
      <td>323</td>
      <td>90</td>
      <td>11</td>
      <td>45</td>
      <td>48</td>
      <td>0</td>
      <td>...</td>
      <td>56</td>
      <td>0</td>
      <td>1</td>
      <td>171</td>
      <td>19</td>
      <td>1049.921053</td>
      <td>133.348684</td>
      <td>56</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8458529</td>
      <td>88</td>
      <td>982.227273</td>
      <td>18</td>
      <td>21</td>
      <td>172</td>
      <td>21</td>
      <td>7</td>
      <td>5</td>
      <td>50</td>
      <td>4</td>
      <td>...</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>-7</td>
      <td>808.840909</td>
      <td>6.965909</td>
      <td>12</td>
      <td>0</td>
      <td>30.769231</td>
    </tr>
    <tr>
      <td>8458537</td>
      <td>258</td>
      <td>1019.697674</td>
      <td>61</td>
      <td>134</td>
      <td>527</td>
      <td>81</td>
      <td>18</td>
      <td>55</td>
      <td>70</td>
      <td>66</td>
      <td>...</td>
      <td>155</td>
      <td>0</td>
      <td>0</td>
      <td>55</td>
      <td>21</td>
      <td>812.027132</td>
      <td>1.325581</td>
      <td>73</td>
      <td>0</td>
      <td>39.285714</td>
    </tr>
    <tr>
      <td>8458590</td>
      <td>179</td>
      <td>931.262570</td>
      <td>34</td>
      <td>32</td>
      <td>325</td>
      <td>185</td>
      <td>8</td>
      <td>10</td>
      <td>88</td>
      <td>28</td>
      <td>...</td>
      <td>36</td>
      <td>1</td>
      <td>1</td>
      <td>87</td>
      <td>-9</td>
      <td>759.122905</td>
      <td>67.363128</td>
      <td>18</td>
      <td>2</td>
      <td>41.791045</td>
    </tr>
    <tr>
      <td>8458938</td>
      <td>163</td>
      <td>1196.533742</td>
      <td>7</td>
      <td>41</td>
      <td>194</td>
      <td>143</td>
      <td>2</td>
      <td>13</td>
      <td>123</td>
      <td>0</td>
      <td>...</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>355</td>
      <td>13</td>
      <td>1009.042945</td>
      <td>101.754601</td>
      <td>15</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8458943</td>
      <td>65</td>
      <td>1014.615385</td>
      <td>12</td>
      <td>27</td>
      <td>116</td>
      <td>64</td>
      <td>3</td>
      <td>9</td>
      <td>24</td>
      <td>17</td>
      <td>...</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>5</td>
      <td>819.969231</td>
      <td>1.292308</td>
      <td>12</td>
      <td>0</td>
      <td>45.945946</td>
    </tr>
    <tr>
      <td>8459053</td>
      <td>175</td>
      <td>1202.914286</td>
      <td>5</td>
      <td>30</td>
      <td>217</td>
      <td>274</td>
      <td>1</td>
      <td>2</td>
      <td>110</td>
      <td>1</td>
      <td>...</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>155</td>
      <td>24</td>
      <td>1027.645714</td>
      <td>141.737143</td>
      <td>3</td>
      <td>0</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>8459424</td>
      <td>63</td>
      <td>1349.619048</td>
      <td>5</td>
      <td>32</td>
      <td>135</td>
      <td>56</td>
      <td>4</td>
      <td>18</td>
      <td>54</td>
      <td>0</td>
      <td>...</td>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>141</td>
      <td>8</td>
      <td>917.968254</td>
      <td>205.857143</td>
      <td>22</td>
      <td>1</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8459427</td>
      <td>71</td>
      <td>765.535211</td>
      <td>5</td>
      <td>14</td>
      <td>80</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>129</td>
      <td>...</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>-8</td>
      <td>709.816901</td>
      <td>48.000000</td>
      <td>0</td>
      <td>0</td>
      <td>46.739130</td>
    </tr>
    <tr>
      <td>8459429</td>
      <td>145</td>
      <td>888.668966</td>
      <td>34</td>
      <td>31</td>
      <td>311</td>
      <td>66</td>
      <td>10</td>
      <td>12</td>
      <td>66</td>
      <td>836</td>
      <td>...</td>
      <td>32</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>7</td>
      <td>750.558621</td>
      <td>3.482759</td>
      <td>22</td>
      <td>0</td>
      <td>50.270595</td>
    </tr>
    <tr>
      <td>8459442</td>
      <td>261</td>
      <td>1053.302682</td>
      <td>45</td>
      <td>94</td>
      <td>347</td>
      <td>164</td>
      <td>8</td>
      <td>21</td>
      <td>150</td>
      <td>2328</td>
      <td>...</td>
      <td>127</td>
      <td>1</td>
      <td>3</td>
      <td>104</td>
      <td>6</td>
      <td>840.793103</td>
      <td>108.582375</td>
      <td>29</td>
      <td>4</td>
      <td>51.848552</td>
    </tr>
    <tr>
      <td>8459454</td>
      <td>80</td>
      <td>630.387500</td>
      <td>1</td>
      <td>2</td>
      <td>55</td>
      <td>73</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>63</td>
      <td>...</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>77</td>
      <td>-16</td>
      <td>529.350000</td>
      <td>98.887500</td>
      <td>0</td>
      <td>1</td>
      <td>44.680851</td>
    </tr>
    <tr>
      <td>8459457</td>
      <td>144</td>
      <td>951.277778</td>
      <td>15</td>
      <td>42</td>
      <td>285</td>
      <td>170</td>
      <td>1</td>
      <td>6</td>
      <td>77</td>
      <td>21</td>
      <td>...</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>-10</td>
      <td>775.534722</td>
      <td>78.770833</td>
      <td>7</td>
      <td>0</td>
      <td>35.593220</td>
    </tr>
    <tr>
      <td>8459461</td>
      <td>105</td>
      <td>927.228571</td>
      <td>13</td>
      <td>41</td>
      <td>132</td>
      <td>30</td>
      <td>3</td>
      <td>11</td>
      <td>28</td>
      <td>573</td>
      <td>...</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
      <td>39</td>
      <td>8</td>
      <td>731.742857</td>
      <td>81.380952</td>
      <td>14</td>
      <td>2</td>
      <td>49.567474</td>
    </tr>
    <tr>
      <td>8459462</td>
      <td>67</td>
      <td>1174.029851</td>
      <td>7</td>
      <td>21</td>
      <td>122</td>
      <td>86</td>
      <td>4</td>
      <td>9</td>
      <td>34</td>
      <td>0</td>
      <td>...</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
      <td>2</td>
      <td>899.298507</td>
      <td>57.447761</td>
      <td>13</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8459492</td>
      <td>159</td>
      <td>1020.603774</td>
      <td>37</td>
      <td>71</td>
      <td>311</td>
      <td>140</td>
      <td>9</td>
      <td>23</td>
      <td>76</td>
      <td>82</td>
      <td>...</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>72</td>
      <td>-4</td>
      <td>834.566038</td>
      <td>1.761006</td>
      <td>32</td>
      <td>0</td>
      <td>51.572327</td>
    </tr>
    <tr>
      <td>8459514</td>
      <td>178</td>
      <td>545.769663</td>
      <td>9</td>
      <td>22</td>
      <td>142</td>
      <td>233</td>
      <td>0</td>
      <td>1</td>
      <td>231</td>
      <td>408</td>
      <td>...</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>1</td>
      <td>469.595506</td>
      <td>66.780899</td>
      <td>1</td>
      <td>0</td>
      <td>55.585831</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 22 columns</p>
</div>



Normally for machine learning methods, we can do some form of _feature selection_ to use the most relevant variables. I am going to do the slightly naive approach of using every possible variable for prediction. This is something that may be done in practice, though it's not the most clever idea; variables can be correlated. Here I'm going to do what's called a "pairplot":


```python
def pairplot(columns, names):
    n_col = columns.shape[1]
    fig, ax = plt.subplots(n_col, n_col)
    
    short_names = {
        'timeOnIce': "time", 
        'goals': "goals", 
        'assists': "assists", 
        'shots': "shots", 
        'hits': "hits", 
        'penaltyMinutes': "PIM", 
        'powerPlayPoints': "PPP", 
        'shortHandedPoints': "SHP"
    }
    
    # Upper-triangular matrix shows correlation between variables
    for i in range(0, n_col-1):
        for j in range(i+1, n_col):
            ax[i,j].scatter(columns[:,i], columns[:,j])
            if j - i > 0:
                ax[i,j].get_yaxis().set_ticklabels([])
                ax[i,j].get_xaxis().set_ticklabels([])
                
            if i == 0:
                ax[i,j].set_title("{}".format(short_names[names[j]]))
            if j == n_col-1:
                ax[i,j].yaxis.set_label_position("right")
                ax[i,j].set_ylabel("{}".format(short_names[names[i]]))

    # Diagonal contains histograms
    for i in range(0, n_col):
        for j in range(0, n_col):
            if i != j: continue
            ax[i,j].hist(columns[:,i], color = '#ffd700')
            
            if i == 0:
                ax[i,j].set_title("{}".format(short_names[names[j]]))
            elif j == (n_col-1):
                ax[i,j].set_xlabel("{}".format(short_names[names[j]]))
    
    # Lower-triangular matrix is hidden
    for i in range(1, n_col):
        for j in range(0, i):
            ax[i,j].axis("off")

    return fig, ax

columns = ['timeOnIce', 'goals', 'assists', 'shots', 'hits', 'penaltyMinutes', 'powerPlayPoints', 'shortHandedPoints']

fig, ax = pairplot(training_set[columns].values, columns)
fig.set_size_inches((10,10))
```


![png](../../../../../assets/notebooks/hockey/output_18_0.png)



```python
# Get the names of the players
train_skaters = skaters[skaters['player_id'].isin(training_ids)].copy()
test_skaters  = skaters[skaters['player_id'].isin(test_ids)].copy()

# Create a dictionary of player IDs to positions, this makes label creation easier
train_position = dict(train_skaters[['player_id','primaryPosition']].values)
test_position = dict(test_skaters[['player_id','primaryPosition']].values)


# Get "labels" which are the hockey players' positions.
train_labels = [train_position[pid] for pid in training_set.index.get_level_values('player_id')]
test_labels  = [test_position[pid] for pid in test_set.index.get_level_values('player_id')]
```

## The "ML bit"

For this exercise, I am going to use the following supervised learning methods; below is a summary along with some pros and cons of each method. I've also tried to write equations where appropriate.

* **Logistic Regression** – Applies the logistic (binary classes) or softmax (multiple) function to a linear combination of weighted variables to predict the probability of class membership.
    * Pros: Model is fairly simple to interpret, with flexibility for regularisation$$\dagger$$.
    * Cons: Assumes a linear relationship between features (after logistic transformation) to class membership
    * $$Pr(Y = c) = \dfrac{ e^{z_c}}{\sum_{i=1}^C e^{z_i} } ~~\mathrm{where}~~ z_i = w_iX+b_i.$$ 
* **Naive Bayes Classifier** – applies "Bayes' rule" to estimate the probability of belonging to a class.
    * Pros: Typically shows good performance and is inexpensive to run.
    * Cons: Assumes that each feature is independent of another
    * $$Pr(Y = c|x_1, x_2... x_n) \propto P(c) \prod_{i=1}^{C} Pr(x_i|c)$$
* **Random Forest Classifier** – bootstraps$\ddagger$ the dataset to create a series of decision trees (the "forest"). New data is then predicted according to all the decision trees, and we take the average prediction. In the case of classification, we take the majority vote.
    * Pros: Possible to trace the importance of specific features using the Gini index; very stable performance.
    * Cons: Difficult to trace how the decision trees were made.
    * For regression, $$\hat{f} = \dfrac{1}{T} \sum_{i=1}^{T} f_i(X_{test})$$
* **Support Vector machines** – finds a hyperplane that best separates classes in a dataset.
    * Pros: coupled with a kernel function, can be applicable for non-linear datasets
    * Cons: sometimes a "soft" margin is required

$$\dagger$$: "regularisation" is a technique where the weights of some terms are shrunk; examples include Lasso and Ridge.

$$\ddagger$$: "bootstrap" here refers to statistical bootstrapping where we sample _with_ replacement.


```python
# Let's get some classifiers
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB # assumes that P(x_i |y) is a Gaussian distribution
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
```

For any classification, we need some mechanism of calculating the performance of our models. There are many measures one can use, but for this exercise, we will simply calculate the accuracy, which is likely the easiest to interpret in this type of whistle-tour blog post. For a pretty visualisation, I will plot the predictions in what's called a "confusion matrix", which shows the distribution of predictions vs. the true answers.


```python
from sklearn.metrics import confusion_matrix
from matplotlib import cm

def accuracy(true, pred):
    assert pred.shape == true.shape, "Shape of pred and true arrays should be the same!"
    return (pred == true).sum() / pred.shape[0]

def get_confusion_matrix(true,pred):
    label_list = list(set(pred) | set(true))
    return confusion_matrix(pred,true,labels=label_list), label_list

def plot_confusion_matrix(cmat, labels, cmap = cm.Greens):
    """
    Plot a heatmap
    """
    fig, ax = plt.subplots()
    ax.imshow(cmat, cmap = cmap)
    
    n_labels = len(labels)
    ticklocs = np.arange(n_labels)
    
    ax.set_xticks(ticklocs)
    ax.set_yticks(ticklocs)
    
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    ax.set_xlim(min(ticklocs)-0.5, max(ticklocs)+0.5)
    ax.set_ylim(min(ticklocs)-0.5, max(ticklocs)+0.5)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    color_threshold = np.max(cmat) * 0.75
    
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            value = cmat[i,j]
            if value >= color_threshold:
                ax.text(j, i, cmat[i,j], color = 'white', ha = 'center', va = 'center')
            else:
                ax.text(j, i, cmat[i,j], ha = 'center', va = 'center')
    
    return fig, ax
```

### Logistic Regression
For the purpose of this exercise I am going to use the (default) logistic regression with the $$l_2$$ penalty
(also known as Ridge regression). I won't go into too many of the mathematical details here but an important
hyper-parameter of the method is the regularisation strength, $$\lambda$$. The higher the value of $$\lambda$$,
this ultimately shrinks the weights closer to 0.


```python
lm = LogisticRegression(solver='lbfgs',multi_class='multinomial', C = 0.1)
lm.fit(training_set.values, train_labels)
```
    LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='multinomial', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
pred = lm.predict(test_set.values)
print("Accuracy of logistic regression is: {}".format(accuracy(np.array(pred), np.array(test_labels))))
```

    Accuracy of logistic regression is: 0.7105263157894737



```python
# plot the confusion matrix
cmat, label_list = get_confusion_matrix(test_labels,pred)
fig, ax = plot_confusion_matrix(cmat, label_list)
plt.show()
```


![png](../../../../../assets/notebooks/hockey/output_27_0.png)



```python
# Plot pred vs. true in a PCA plot
from sklearn.decomposition import PCA

def pca_plot(pred, method):

    # scale the data
    tv = (test_set - test_set.mean()) / test_set.std()

    pca = PCA()
    new_data = pca.fit_transform(tv)

    colors = {
        "RW": "#ffd700",
        "D": "#1348ae",
        "C": "#90ee90",
        "LW": "#e8291c"
    }

    true_labels_to_colors = [ colors[pos] for pos in test_labels ]
    pred_labels_to_colors = np.array([ colors[pos] for pos in pred ])

    fig, ax = plt.subplots(1,2, sharey=True)
    ax[0].scatter(new_data[:,0], new_data[:,1], 
                  alpha = 0.5, 
                  color = true_labels_to_colors)
    
    # 
    for lab in set(pred):
        pos_idx = np.argwhere(pred == lab).flatten()
        ax[1].scatter(new_data[pos_idx,0], new_data[pos_idx,1], 
                      color = pred_labels_to_colors[pos_idx], 
                      alpha = 0.5,
                      label = lab)
    
    
    ax[0].set_title("True labels")
    ax[1].set_title("Predicted labels")
    
    ax[1].legend(loc = 'upper left', ncol = 2)
    
    fig.suptitle(method)
    fig.set_size_inches((10,5))
    
    return fig, ax

fig, ax = pca_plot(pred, "Logistic Regression")
```


![png](../../../../../assets/notebooks/hockey/output_28_0.png)


### Random Foest
As before, I will just use the default implementation.

```python
# let's train a "simple" random forest
rf = RandomForestClassifier()
```


```python
rf.fit(training_set.values, train_labels)
```
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
pred = rf.predict(test_set.values)
print("Accuracy of random forest is: {}".format(accuracy(np.array(pred), np.array(test_labels))))
```

    Accuracy of random forest is: 0.6929824561403509



```python
cmat, label_list = get_confusion_matrix(test_labels, pred)
fig, ax = plot_confusion_matrix(cmat, label_list)
```


![png](../../../../../assets/notebooks/hockey/output_33_0.png)



```python
fig, ax = pca_plot(pred, "Random Forest")
```


![png](../../../../../assets/notebooks/hockey/output_34_0.png)


### Naive Bayes Classifier
For the NBC, I will again use the default implementation but assume that every variable has a Gaussian
distribution. This is not ideal by any means, but is easiest to code and gives you a flavour of what it does.

```python
# let's train a "simple" naive bayes classifier
nbc = GaussianNB()
```


```python
nbc.fit(training_set.values, train_labels)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
pred = nbc.predict(test_set.values)
print("Accuracy of Naive Bayes classifier is: {}".format(accuracy(np.array(pred), np.array(test_labels))))
```

    Accuracy of Naive Bayes classifier is: 0.5847953216374269



```python
fig, ax = pca_plot(pred, "NBC")
```


![png](../../../../../assets/notebooks/hockey/output_39_0.png)


### Support Vector Machines
Here I will use the `LinearSVC` class; essentially we are applying a linear kernel to the data. What this
means is that essentially we are assuming that no transformation is needed to draw a hyperplane that will
separate the data.

```python
svc = LinearSVC(max_iter=2000)
```


```python
svc.fit(training_set.values, train_labels)
```

    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=2000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0)




```python
pred = svc.predict(test_set.values)
print("Accuracy of SVC is: {}".format(accuracy(np.array(pred), np.array(test_labels))))
```

    Accuracy of SVC is: 0.7017543859649122



```python
cmat, label_list = get_confusion_matrix(test_labels, pred)
fig, ax = plot_confusion_matrix(cmat, label_list)
plt.show()
```


![png](../../../../../assets/notebooks/hockey/output_44_0.png)



```python
fig, ax = pca_plot(pred, "SVM")
```


![png](../../../../../assets/notebooks/hockey/output_45_0.png)


## Revision

No method was a true outstanding performer. While the random forest classifier did have the highest level of accuracy, it was only marginally better than logistic regression.

It would be worth seeing why certain methods failed to classify a player into the correct primary position. We could go more in-depth and ask,
* Is this a case where we over-penalise ourselves (e.g. left-wing vs. right-wing players are not that different)?
* Is this a case where a player has out-of-position behaviours (e.g. a defenseman with some high goals/assists? a forward who is a defensive specialist?)
* Is there not enough game data?

Going further, we can ask...
* Are there fundamental aspects of the ML methods tested here that make it unsuitable for this problem?
* Can we do feature selection of some sort?
* What other information can we get to improve prediction? For example, does stick handed-ness have any bearing on position?


```python
from scipy.stats import gaussian_kde

test_set_copy = test_set.copy()
test_set_copy['pred'] = rf.predict(test_set_copy)

test_to_names = pd.merge(
    left = test_set_copy,
    right = skaters,
    how = 'inner', on = 'player_id'
)

correct = test_to_names[test_to_names['primaryPosition']==test_to_names['pred']].copy()
incorrect = test_to_names[test_to_names['primaryPosition']!=test_to_names['pred']].copy()

incorrect[['firstName', 'lastName', 'primaryPosition', 'pred']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>firstName</th>
      <th>lastName</th>
      <th>primaryPosition</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Teemu</td>
      <td>Selanne</td>
      <td>RW</td>
      <td>LW</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Todd</td>
      <td>Bertuzzi</td>
      <td>RW</td>
      <td>LW</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Mike</td>
      <td>Grier</td>
      <td>RW</td>
      <td>C</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Ryan</td>
      <td>Smyth</td>
      <td>LW</td>
      <td>C</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Richard</td>
      <td>Park</td>
      <td>RW</td>
      <td>C</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>327</td>
      <td>Nick</td>
      <td>Lappin</td>
      <td>RW</td>
      <td>LW</td>
    </tr>
    <tr>
      <td>328</td>
      <td>Patrik</td>
      <td>Laine</td>
      <td>RW</td>
      <td>LW</td>
    </tr>
    <tr>
      <td>334</td>
      <td>Vinni</td>
      <td>Lettieri</td>
      <td>C</td>
      <td>LW</td>
    </tr>
    <tr>
      <td>340</td>
      <td>Brady</td>
      <td>Tkachuk</td>
      <td>LW</td>
      <td>RW</td>
    </tr>
    <tr>
      <td>341</td>
      <td>Dominik</td>
      <td>Kahun</td>
      <td>C</td>
      <td>LW</td>
    </tr>
  </tbody>
</table>
<p>105 rows × 4 columns</p>
</div>




```python
correct_gp = correct['games_played'].values
incorrect_gp = incorrect['games_played'].values

# We can create a Gaussian kernel on top of the number of games played to do some comparisons
correct_kde = gaussian_kde(correct_gp)
incorrect_kde = gaussian_kde(incorrect_gp)

num_games = np.arange(1, 801)

fig, ax = plt.subplots(1,1)
ax.plot(num_games, correct_kde.evaluate(num_games), color = '#134a8e', label = "Correct predictions")
ax.plot(num_games, incorrect_kde.evaluate(num_games), color = '#e8291c', label = "Incorrect predictions")
ax.set_xlabel("Number of games played")
ax.set_ylabel("Density")
```




    Text(0, 0.5, 'Density')




![png](/assets/notebooks/hockey/output_48_1.png)


What's interesting here is that:
* For the random forest, mis-classifications are only found for forwards (no defensemen are ever classified as forwards and vice-versa).
* There are more winger mis-classifications (actual = RW, predicted = LW), which may imply a too-stringent classification scheme.
* This doesn't seem to be affected by the number of games played by the players as they have similar distributions.

While we can explore the data further to explain misclassifications, I think that's outside the scope of this post and that's for next time...
