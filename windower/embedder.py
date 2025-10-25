from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import random, torch, os
from visualize import reduce_and_plot


def embed_and_plot(model, seed, training_epochs):
    
    random.seed(seed)
    torch.manual_seed(seed)

    input_dir = "KGs"
    os.makedirs(input_dir, exist_ok=True)

    print("Loading all KG TSV files...")

    tsv_files = [f for f in os.listdir(input_dir) if f.endswith(".tsv")]

    if not tsv_files:
        raise FileNotFoundError("No .tsv files found in ./KGs directory.")

    for tsv_file in tsv_files:
        base_name = os.path.splitext(tsv_file)[0]
        path = os.path.join(input_dir, tsv_file)

        print(f"\n--- Processing {base_name} ---")

        tf = TriplesFactory.from_path(path)
        training, testing, validation = tf.split([0.8, 0.1, 0.1])

        print(f"Training {model}...")
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=model,
            random_seed=seed,
            training_kwargs=dict(num_epochs=training_epochs),
        )

        print(f"Training complete for {base_name}!")

        reduce_and_plot(result=result, basename=f"{base_name}_{model}", only_entities=True)
        
embed_and_plot("MurE", 42, 100)

#print(result.model.entity_representations[0])

#for n,i in result.training.entity_to_id.items(): 
    #print(f"{n:15s} →", result.model.entity_representations[0]().detach()[i].numpy())

# Drop into interactive shell
#import code
#code.interact(local=locals())
