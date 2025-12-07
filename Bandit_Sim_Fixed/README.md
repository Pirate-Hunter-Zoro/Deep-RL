# Bandit_Sim

A python based multi-armed bandit simulator for CS5313/7313. 

### Author

Robert Geraghty
Altered by Jacob Brue



## About

This simulator generates an n armed bandit, with each arm having a clipped normally distributed payout linearly increasing means. Every arm shares a common standard deviation that is passed as a parameter.

## Requirements

This code makes use of `Matplotlib` and `Numpy`

## Usage

First, to initialize a simulator:

```python
bandsim = Bandit_Sim(n_arms, payout_std, clip=clip_stdevs, cseed=seed)
```
If you want to veiw the histograms of all the arms, use the `plot` method to display two plots. This is done by:
```python
bandsim.plot(num_samples)
```
Where `num_samples` is the number of samples generated for each arm. The first plot generated shows the histogram of the data for each arm combined into one joint histogram.
The second plot shows multiple histograms, one for each arm of the bandit. This method can be useful to get a sense of the problem given your parameters.

To actually pull an arm of the bandit, you use:
```python
payout = bandsim.pull_arm(n)
```
where `n` is the index of the arm you want to pull. This method will return a sample payout from the chosen arm's distribution.

Finally, the arm means can be accessed with:
```python
bandsim.arm_means
```

## Support

If you find a bug please email me at jacob-brue@utulsa.edu.

