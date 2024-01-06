report_system_prompt = (
    "You are a helpful assistant. Always answer in the most accurate way."
)
story_system_prompt = (
    "You are a helpful assistant. Always answer in the most accurate way."
)
fake_news_system_prompt = "You are a helpful assistant. Always respond with realistic yet invented articles."

report_prompt = "Write a book report about '{}', written by {}."
story_prompt = "Write a {}story about {}."
fake_news_prompt = "Write a news article about {}'s visit to {} in {}."

report_topics = [
    ("Pride and Prejudice", "Jane Austen"),
    ("Persuasion", "Jane Austen"),
    ("Emma", "Jane Austen"),
    ("Don Quixote", "Cervantes"),
    ("The Lord of the Rings", "Tolkien"),
    ("The Hobbit", "Tolkien"),
    ("And Then There Were None", "Agatha Cristie"),
    ("Alice's Adventures in Wonderland", "Lewis Carroll"),
    ("Catcher in the Rye", "Salinger"),
    ("In Search of Lost Time", "Marcel Proust"),
    ("Ulysses", "James Joyce"),
    ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
    ("Love in the Time of Cholera", "Gabriel Garcia Marquez"),
    ("The Great Gatsby", "F. Scott Fitzgerald"),
    ("Tender Is the Night", "F. Scott Fitzgerald"),
    ("Moby Dick", "Herman Melville"),
    ("War and Peace", "Leo Tolstoy"),
    ("The Call of the Wild", "Jack London"),
    ("Hamlet", "William Shakespeare"),
    ("Twelfth Night", "William Shakespeare"),
    ("Macbeth", "William Shakespeare"),
    ("Romeo and Juliet", "William Shakespeare"),
    ("The Tempest", "William Shakespeare"),
    ("King Lear", "William Shakespeare"),
    ("The Odyssey", "Homer"),
    ("Madame Bovary", "Gustave Flaubert"),
    ("The Divine Comedy", "Dante Alighieri"),
    ("The Brothers Karamazov", "Fyodor Dostoyevsky"),
    ("Crime and Punishment", "Fyodor Dostoyevsky"),
    ("The Idiot", "Fyodor Dostoyevsky"),
    ("The Possessed", "Fyodor Dostoyevsky"),
    ("Wuthering Heights", "Emily Brontë"),
    ("One Flew Over the Cuckoo's Nest", "Ken Kesey"),
    ("The Adventures of Huckleberry Finn", "Mark Twain"),
    ("Anna Karenina", "Leo Tolstoy"),
    ("The Iliad", "Homer"),
    ("To the Lighthouse", "Virginia Woolf"),
    ("Catch-22", "Joseph Heller"),
    ("Heart of Darkness", "Joseph Conrad"),
    ("The Sound and the Fury", "William Faulkner"),
    ("Nineteen Eighty Four", "George Orwell"),
    ("Animal Farm", "George Orwell"),
    ("Great Expectations", "Charles Dickens"),
    ("David Copperfield", "Charles Dickens"),
    ("A Tale of Two Cities", "Charles Dickens"),
    ("Oliver Twist", "Charles Dickens"),
    ("The Grapes of Wrath", "John Steinbeck"),
    ("Of Mice and Men", "John Steinbeck"),
    ("Absalom, Absalom!", "William Faulkner"),
    ("Invisible Man", "Ralph Ellison"),
    ("To Kill a Mockingbird", "Harper Lee"),
    ("The Trial", "Franz Kafka"),
    ("The Metamorphosis", "Franz Kafka"),
    ("The Castle", "Franz Kafka"),
    ("The Red and the Black", "Stendhal"),
    ("The Charterhouse of Parma", "Stendhal"),
    ("Middlemarch", "George Eliot"),
    ("Gulliver's Travels", "Jonathan Swift"),
    ("Beloved", "Toni Morrison"),
    ("Mrs. Dalloway", "Virginia Woolf"),
    ("The Waves", "Virginia Woolf"),
    ("The Stranger", "Albert Camus"),
    ("The Plague", "Albert Camus"),
    ("The Myth of Sisyphus", "Albert Camus"),
    ("Jane Eyre", "Charlotte Bronte"),
    ("Vilette", "Charlotte Bronte"),
    ("The Aeneid", "Virgil"),
    ("The Sun Also Rises", "Ernest Hemingway"),
    ("The Old Man and the Sea", "Ernest Hemingway"),
    ("A Farewell to Arms", "Ernest Hemingway"),
    ("Candide", "Voltaire"),
    ("Zadig", "Voltaire"),
    ("Micromegas", "Voltaire"),
    ("Les Miserables", "Victor Hugo"),
    ("Frankenstein", "Mary Shelley"),
    ("Antigone", "Sophocles"),
    ("Electra", "Sophocles"),
    ("Lord of the Flies", "William Golding"),
    ("Brave New World", "Aldous Huxley"),
    ("Journey to the End of The Night", "Celine"),
    ("A Sentimental Education", "Gustave Flaubert"),
    ("The Handmaid's Tale", "Margaret Atwood"),
    ("Charlotte's Web", "E. B. White"),
    ("Gargantua and Pantagruel", "Francois Rabelais"),
    ("Faust", "Goethe"),
    ("Robinson Crusoe", "Daniel Defoe"),
    ("A Clockwork Orange", "Anthony Burgess"),
    ("The Master and Margarita", "Mikhail Bulgakov"),
    ("Father Goriot", "Honore de Balzac"),
    ("Cousin Bette", "Honore de Balzac"),
    ("The Human Comedy", "Honore de Balzac"),
    ("The Little Prince", "Antoine de Saint-Exupéry"),
    ("The Count of Monte Cristo", "Alexandre Dumas"),
    ("The Lion, The Witch and the Wardrobe", "C. S. Lewis"),
    ("Twenty Thousand Leagues Under the Sea", "Jules Verne"),
    ("The Wind-Up Bird Chronicle", "Haruki Murakami"),
    ("Fahrenheit 451", "Ray Bradbury"),
    ("Harry Potter And The Philosopher's Stone", "J. K Rowling"),
    ("Dune", "Frank Herbert"),
    ("The Three-Body Problem", "Liu Cixin"),
]


t1 = ["", "funny ", "sad ", "dramatic ", "suspenseful ", "thrilling "]
t2 = [
    "a man on a quest to find the Holy Grail.",
    "two college friends falling in love.",
    "a policeman saving a building held hostage by group of terrorists.",
    "the struggle of publishing an academic paper.",
    "a murder investigation in an old mansion.",
    "a young prodigy that becomes orphaned.",
    "a middle-aged woman that discovers a ghost and befriends it.",
    "a long journey to Japan that is interrupted by a disaster.",
    "a poor child that comes into an unexpected fortune.",
    "three strangers that win a getaway vacation together.",
    "a retired astronaut that joins a risky interstellar rescue mission.",
    "an AI that begins to question its own existence.",
    "a small coastal town plagued by inexplicable supernatural occurrences.",
    "a reclusive writer that receives mysterious, prophetic letters in the mail.",
    "a linguist tasked with deciphering an ancient language that holds the secrets of a lost civilization.",
    "an antique restorer that finds an enchanted mirror showing glimpses of different timelines.",
]


story_topics = [(i, j) for i in t1 for j in t2][:100]

person = [
    "Narendra Modi",
    "Barack Obama",
    "Denis Sassou Nguesso",
    "Emmanuel Macron",
    "Fumio Kishida",
    "Angela Merkel",
    "Kim Jong Un",
    "Justin Trudeau",
]
location = [
    "a peace conference",
    "an international summit",
    "a diplomatic event",
    "the summer olympics",
]

fake_news_topics = [
    (person[i], person[j], location[k])
    for i in range(len(person))
    for j in range(len(person))
    for k in range(len(location))
    if j > i
][:100]


raw_prompts = [
    (raw_prompt.format(*topic), sys_prompt)
    for raw_prompt, sys_prompt, topics in [
        (report_prompt, report_system_prompt, report_topics),
        (story_prompt, story_system_prompt, story_topics),
        (fake_news_prompt, fake_news_system_prompt, fake_news_topics),
    ]
    for topic in topics
]
