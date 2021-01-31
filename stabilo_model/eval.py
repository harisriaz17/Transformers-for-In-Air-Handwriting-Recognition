import pandas as pd

target_list = []
preds_list = []
for i in range(X_val.shape[0]):
  val_probas, val_targets, val_preds = learn.get_X_preds(X_val[i], y_val[i], with_decoded=True)
  decoded_target = learn.decoder(val_targets)
  decoded_pred = learn.decoder(torch.argmax(val_preds, dim=1))
  target_list.append(decoded_target[0])
  preds_list.append(decoded_pred[0])


val_df = pd.DataFrame(list(zip(target_list, preds_list)), columns =['Actual letter', 'Predicted Letter'])
val_df.style.set_properties(**{'text-align': 'left'})
display_df(val_df)
print("Validation set accuracy (Case-sensitive):", 100 * (sum(1 for x, y in zip(target_list, preds_list) if x == y) / len(target_list)), "%.")
print("Validation set accuracy (Case-insensitive):", 100 * (sum(1 for x, y in zip(target_list, preds_list) if x.lower() == y.lower()) / len(target_list)), "%.")