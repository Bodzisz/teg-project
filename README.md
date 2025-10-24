### TEG Project

[Opis projektu](https://github.com/wodecki/TEG_2025/blob/main/src/3.%20Retrieval%20Augmented%20Generation/07_Your_Project_TalentMatch/PRD.md)

## Docker

### Przydatne komendy Dockera

Otwórz terminal w tym samym katalogu, w którym znajduje się plik `docker-compose.yml`, i uruchom te polecenia.

- **Uruchom kontener:**

  ```bash
  docker-compose up -d
  ```

  Flaga `-d` uruchamia go w trybie "detached" (w tle).

- **Sprawdź działające kontenery:**

  ```bash
  docker ps
  ```

  Powinieneś zobaczyć na liście swój kontener `neo4j`. Flaga `a` pokaże wszystkie kontenery, w tym zatrzymane.

- **Wyświetl logi kontenera:**

  ```bash
  docker-compose logs -f neo4j
  ```

  To polecenie pokazuje na żywo dane wyjściowe z serwera Neo4j, co jest przydatne do debugowania. Naciśnij `Ctrl+C`, aby zatrzymać śledzenie.

- **Zatrzymaj i usuń kontener:**

  ```bash
  docker-compose down
  ```

  To polecenie zatrzymuje i usuwa kontener, ale Twoje dane są bezpieczne, ponieważ użyłeś wolumenów.

---

## Neo4j

1.  **Uzyskaj dostęp do Neo4j Browser:**
    Gdy kontener jest uruchomiony, otwórz przeglądarkę internetową i przejdź pod adres **`http://localhost:7474`**.

2.  **Zaloguj się:**
    Zobaczysz ekran logowania. Użyj passów, które ustawiłeś w pliku `docker-compose.yml` i jesteś gotów do uruchamiania zapytań Cypher\!

### Uruchamianie poleceń wewnątrz kontenera

Czasami trzeba uruchomić polecenia administracyjne bezpośrednio.

1.  **Wejdź do powłoki kontenera:**
    To polecenie daje Ci dostęp do wiersza poleceń `bash` wewnątrz działającego kontenera `neo4j`.

    ```bash
    docker exec -it neo4j bash
    ```

2.  **Użyj narzędzi Neo4j:**
    Teraz, gdy jesteś "w środku" kontenera, możesz używać wbudowanych narzędzi wiersza poleceń Neo4j.

    - **Sprawdź status bazy danych:** Użyj narzędzia `neo4j-admin`, aby sprawdzić, czy baza danych działa.

      ```bash
      neo4j-admin dbms status
      ```

    - **Użyj Cypher Shell:** To interfejs wiersza poleceń do uruchamiania zapytań.

      ```bash
      # Uruchom powłokę i zaloguj się
      cypher-shell -u neo4j -p password

      # Wykonaj twoje zapytanie Cypher

      # Aby wyjść z Cypher Shell, wpisz:
      :exit
      ```
