# Lepus Classifier

## Overview

This project aims to distinguish between the Eurpoean Hare and the Eastern Cotton-tail Rabbit class
using a relatively small set of images retrieved online.

## Data

Data was retrieved and located in [resources/data.txt](resources/data.txt)

## Quickstart

1. Install dependencies.
```
  pip install -r requirements.txt
```

2. Login to your Weights and Biases account using your API key.
```
  wanb login
```

3. Use the bootstrap function to start training your custom model. 
```
import src.bootstrap import bootstrap

bootstrap()
```
