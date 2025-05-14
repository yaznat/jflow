package jflow.model;

/**
 * Interface between the user and the train function to 
 * facilitate the saving of model checkpoionts.
 */
class ModelCheckpoint {
    private String metric;
    private String savePath;
    /**
     * Passes data to the train function to faciliate the saving of model checkpoints.
     * @param metric                            the metric to track for improvement. Supported: 
     *                                            <ul> <li> val_loss <li> val_accuracy 
     *                                                 <li> train_loss <li> train_accuracy </ul>
     * @param savePath                          the path to save checkpoints to.
     */
    protected ModelCheckpoint(String metric, String savePath) {
        this.metric = metric;
        this.savePath = savePath;
    }

    protected String getMetric() {
        return metric;
    }

    protected String getSavePath() {
        return savePath;
    }


}
