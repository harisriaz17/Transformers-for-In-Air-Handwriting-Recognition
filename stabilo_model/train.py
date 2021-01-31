model = TransformerEncoder(dls.vars, dls.c, dls.len, n_layers=3, res_dropout=0.3, fc_dropout=0.5, n_heads=16, d_model=64, d_ff=64, d_v=16, d_k=16)
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy],  cbs=ShowGraphCallback2())
learn.lr_find()

n_epochs = 100
model = TST(dls.vars, dls.c, dls.len, n_layers=3, res_dropout=0.3, fc_dropout=0.5, n_heads=16, d_model=64, d_ff=64, d_v=16, d_k=16)
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy], cbs=ShowGraphCallback2())
start = time.time()

learn.fit(n_epoch=n_epochs, lr=1e-3)
print('\nElapsed time:', time.time() - start)
learn.plot_metrics()
learn.save('initial_transformer.pkl')
