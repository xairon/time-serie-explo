import { Link } from 'react-router-dom'

export default function PrivacyPage() {
  return (
    <div className="flex justify-center p-6">
      <div className="w-full max-w-2xl space-y-6 text-text-secondary text-sm leading-relaxed">
        <div>
          <h1 className="text-xl font-bold text-text-primary">Politique de confidentialité</h1>
          <p className="mt-1 text-text-muted italic">Contenu à valider par le DPO du BRGM.</p>
        </div>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-text-primary">Données traitées</h2>
          <p>
            L'application traite les données de compte (adresse e-mail, nom affiché, rôle) ainsi que
            les données que vous importez et produisez dans l'atelier (jeux de données piézométriques,
            analyses, modèles). Des journaux de connexion (date, adresse IP, type d'événement) sont
            conservés à des fins de sécurité.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-text-primary">Finalité</h2>
          <p>
            Ces données servent à fournir l'accès authentifié à l'application, à exécuter les analyses
            demandées et à assurer la sécurité du service.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-text-primary">Durée de conservation</h2>
          <p>
            Les journaux d'authentification sont conservés 365 jours puis supprimés automatiquement.
            Les données de compte sont conservées tant que le compte est actif.
          </p>
        </section>

        <section className="space-y-2">
          <h2 className="text-base font-semibold text-text-primary">Vos droits</h2>
          <p>
            Vous disposez d'un droit d'accès, de rectification et d'effacement de vos données. Vous
            pouvez supprimer définitivement votre compte et les données associées depuis la page{' '}
            <Link to="/account" className="text-accent-cyan hover:underline">Mon compte</Link>.
            Pour toute autre demande, contactez le délégué à la protection des données (DPO) du BRGM.
          </p>
        </section>
      </div>
    </div>
  )
}
