import build_features
import logging

A = build_features.main()
logging.info('Train size:', A.shape)
#logging.info('Target size:',y.shape)
